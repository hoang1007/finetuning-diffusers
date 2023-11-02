from __future__ import annotations
from typing import Iterable, Optional, Tuple, TYPE_CHECKING
import os
import math
from tqdm import tqdm

import torch
import accelerate
import diffusers

from torch.utils.data import DataLoader, Dataset
from torch.nn import Parameter
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler

from accelerate import DistributedDataParallelKwargs

from mugen.utils.trainer_utils import (
    set_seed,
    get_latest_checkpoint,
    prune_checkpoints,
    is_using_gpu,
)
from mugen.ddp_wrapper import unwrap_model, DDPWrapper
from mugen.hooks import HookHandler

if TYPE_CHECKING:
    from mugen import TrainingArguments
    from mugen.trainingmodules import TrainingModule
    from mugen.datamodules import BaseDataModule
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


logger = get_logger(__name__, log_level="INFO")


class Trainer:
    def __init__(
        self,
        project_name: str,
        training_module: TrainingModule,
        training_args: TrainingArguments,
        data_module: Optional[BaseDataModule] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.training_args = training_args
        self.global_step = 0

        set_seed(self.training_args.seed)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            mixed_precision=training_args.mixed_precision,
            log_with=training_args.logger,
            cpu=training_args.use_cpu,
            deepspeed_plugin=training_args.get_deepspeed_plugin(),
            fsdp_plugin=training_args.get_fsdp_plugin(),
            project_config=training_args.get_project_configuration(),
            kwargs_handlers=[ddp_kwargs],
        )

        if self.accelerator.is_local_main_process:
            diffusers.utils.logging.set_verbosity_info()
        else:
            diffusers.utils.logging.set_verbosity_error()

        self.training_module = self._setup_training_module(training_module)

        num_trainable_params = sum(
            p.numel() for p in training_module.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in training_module.parameters())
        print(f"NUM TRAINABLE PARAMETERS: {num_trainable_params:,}")
        print(f"TOTAL PARAMETERS: {total_params:,}")

        self.train_dataloader, self.val_dataloader = self._setup_data_loader(
            data_module, train_dataset, eval_dataset
        )

        self.optimizers = [
            self._create_optimizer(params)
            for params in unwrap_model(self.training_module).get_optim_params()
        ]

        num_training_steps = len(self.train_dataloader) * self.training_args.num_epochs
        self.schedulers = [
            self._create_scheduler(
                opt,
                num_training_steps=num_training_steps,
                num_warmup_steps=self.training_args.get_warmup_steps(
                    num_training_steps
                ),
            )
            for opt in self.optimizers
        ]

        # Prepare with Accelerator
        prepared = self.accelerator.prepare(
            self.training_module,
            self.train_dataloader,
            self.val_dataloader,
            *self.optimizers,
            *self.schedulers,
        )
        self.training_module = prepared[0]
        self.train_dataloader = prepared[1]
        self.val_dataloader = prepared[2]
        self.optimizers = prepared[3 : 3 + len(self.optimizers)]
        self.schedulers = prepared[3 + len(self.optimizers) :]

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name,
                init_kwargs={
                    self.training_args.logger: self.training_args.tracker_init_kwargs
                },
            )

        self.hook_handler = HookHandler()
        self.hook_handler.register_hook(unwrap_model(self.training_module))

    def start(self):
        total_batch_size = (
            self.training_args.train_batch_size
            * self.accelerator.num_processes
            * self.training_args.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.training_args.gradient_accumulation_steps
        )
        max_train_steps = self.training_args.num_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.training_args.num_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.training_args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")

        first_epoch = 0

        if self.training_args.resume_from_checkpoint:
            path = self.load_state(self.training_args.resume_from_checkpoint)

            if path is None:
                self.training_args.resume_from_checkpoint = None
                logger.info("No checkpoint found. Starting a new training run!")
            else:
                self.global_step = int(os.path.basename(path).split("_")[-1])

                resume_global_step = (
                    self.global_step * self.training_args.gradient_accumulation_steps
                )
                first_epoch = self.global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % num_update_steps_per_epoch

        # Train!
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.hook_handler.on_start()
        progress_bar = tqdm(
            range(max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=True,
        )
        progress_bar.set_description(f"Training...")
        unwrap_model(self.training_module).register_progress_bar(progress_bar)

        for epoch in range(first_epoch, self.training_args.num_epochs):
            self.training_module.train()
            self.hook_handler.on_train_epoch_start()
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                    self.training_args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.training_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                self.hook_handler.on_train_batch_start()

                with self.accelerator.accumulate(self.training_module):
                    for opt_idx, opt in enumerate(self.optimizers):
                        opt.zero_grad()
                        loss = self.training_module(batch, step, opt_idx)
                        self.accelerator.backward(loss)
                        if self.training_args.max_grad_norm is not None:
                            if self.accelerator.sync_gradients:
                                self._clip_grad_norm_(
                                    unwrap_model(
                                        self.training_module
                                    ).get_optim_params()[opt_idx],
                                )
                        opt.step()
                    for scheduler in self.schedulers:
                        scheduler.step()

                if self.accelerator.sync_gradients:
                    self.hook_handler.on_train_batch_end()
                    progress_bar.update(1)
                    self.global_step += 1

                    if (
                        self.global_step % self.training_args.save_steps == 0
                        or self.global_step == max_train_steps
                    ):
                        state_save_dir = os.path.join(
                            self.training_args.output_dir, "checkpoints"
                        )

                        self.accelerator.wait_for_everyone()
                        self.accelerator.save_state(
                            os.path.join(
                                state_save_dir, f"checkpoint_{self.global_step}"
                            )
                        )

                        if self.accelerator.is_main_process:
                            if self.training_args.save_total_limit is not None:
                                prune_checkpoints(
                                    state_save_dir,
                                    self.training_args.save_total_limit - 1,
                                )
                            unwrap_model(self.training_module).save_pretrained(
                                self.training_args.output_dir
                            )

                    if (
                        self.global_step
                        % self.training_args.get_eval_steps(max_train_steps)
                        == 0
                        and self.accelerator.is_main_process
                    ):
                        self._eval_loop()

            if self.accelerator.is_main_process:
                self.hook_handler.on_train_epoch_end()

            self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def _eval_loop(self):
        old_train = self.training_module.training
        self.training_module.eval()
        with tqdm(
            total=len(self.val_dataloader),
            disable=not self.accelerator.is_local_main_process,
            leave=False,
        ) as progress_bar:
            progress_bar.set_description(f"Evaluating...")

            with torch.inference_mode():
                self.hook_handler.on_validation_epoch_start()
                for step, batch in enumerate(self.val_dataloader):
                    self.training_module(batch, step)
                    progress_bar.update(1)

                if self.accelerator.is_main_process:
                    self.hook_handler.on_validation_epoch_end()
        self.training_module.train(old_train)
        if is_using_gpu(self.accelerator):
            self.accelerator.print(
                f"\nGPU memory used for eval: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            )
            torch.cuda.empty_cache()

    def evaluate(self):
        self._eval_loop()

    def get_tracker(self, unwrap: bool = False):
        if self.training_args.logger is not None:
            return self.accelerator.get_tracker(self.training_args.logger, unwrap)

    def _create_optimizer(self, parameters: Iterable[Parameter]):
        if (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            optimizer_cls = accelerate.utils.DummyOptim
        elif self.training_args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam optimizer"
                )
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            parameters,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
            weight_decay=self.training_args.adam_weight_decay,
        )

    def _create_scheduler(
        self, optimizer: Optimizer, num_training_steps: int, num_warmup_steps: int
    ) -> LRScheduler:
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                name=self.training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            lr_scheduler = accelerate.utils.DummyScheduler(
                optimizer,
                total_num_steps=num_training_steps,
                warmup_num_steps=num_warmup_steps,
            )

        return lr_scheduler

    def _get_train_dataloader(self, dataset: Dataset):
        if self.training_args.data_seed is not None:
            generator = torch.Generator().manual_seed(self.training_args.data_seed)
        else:
            generator = None

        return DataLoader(
            dataset,
            batch_size=self.training_args.train_batch_size,
            num_workers=self.training_args.data_loader_num_workers,
            generator=generator,
            shuffle=True,
        )

    def _get_eval_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.training_args.eval_batch_size,
            num_workers=self.training_args.data_loader_num_workers,
            shuffle=False,
        )

    def _clip_grad_norm_(self, parameters: Iterable[torch.nn.Parameter]):
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                parameters, self.training_args.max_grad_norm
            )
            
    def load_state(self, path: str = 'latest'):
        """Load the state of the trainer from a checkpoint.

        Args:
            path (str, optional): Path to the checkpoint directory or `latest` keyword to load the latest checkpoint automatically. Defaults to 'latest'.

        Returns:
            str: Path to the checkpoint directory or `None` if loading state unsuccessfully.
        """
        if path == "latest":
            path = get_latest_checkpoint(
                os.path.join(self.training_args.output_dir, "checkpoints")
            )

        if path is None or not os.path.exists(path):
            self.accelerator.print(
                f"Checkpoint not found at {path}"
            )
            return None
        else:
            self.accelerator.print(f"Loading checkpoint from {path}")
            self.accelerator.load_state(path)
        
        return path

    def _setup_training_module(self, training_module: TrainingModule) -> TrainingModule:
        training_module.register_trainer(self)

        if self.training_args.use_lora:
            from peft import get_peft_model
            from peft import LoHaConfig

            training_module = get_peft_model(
                model=training_module,
                peft_config=LoHaConfig(
                    r=self.training_args.lora_rank,
                    alpha=self.training_args.lora_alpha,
                    rank_dropout=self.training_args.lora_rank_dropout,
                    module_dropout=self.training_args.lora_module_dropout,
                    use_effective_conv2d=self.training_args.use_effective_conv2d,
                    target_modules=training_module.LORA_TARGET_MODULES,
                    init_weights=True,
                ),
            )
        # Wrap TrainingModule with DDPWrapper
        training_module = DDPWrapper(training_module)
        return training_module

    def _setup_data_loader(
        self,
        data_module: BaseDataModule = None,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if (
            train_dataset is not None or eval_dataset is not None
        ) and data_module is not None:
            raise ValueError(
                "The data module is specified. You should not pass the training dataset and evaluation dataset."
            )
        if data_module is None and train_dataset is None and eval_dataset is None:
            raise ValueError(
                "The data module is not specified. You should pass the training dataset or evaluation dataset."
            )

        if data_module is not None:
            if self.accelerator.is_main_process:
                data_module.prepare_data()
            self.accelerator.wait_for_everyone()
            data_module.setup()

            train_dataset = data_module.get_training_dataset()
            eval_dataset = data_module.get_validation_dataset()

        train_dataloader = None
        eval_dataloader = None

        if train_dataset is not None:
            train_dataloader = self._get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_dataloader = self._get_eval_dataloader(eval_dataset)

        return train_dataloader, eval_dataloader
