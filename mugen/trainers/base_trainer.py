from typing import Dict, Literal, List, Union, TypeVar
import os
import math
from tqdm import tqdm

import torch
import accelerate
import diffusers

from torch.utils.data import DataLoader, Dataset
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim import Optimizer

logger = get_logger(__name__, log_level="INFO")
TrainingWrapper_ = TypeVar("TrainingWrapper_", bound="TrainingWrapper")


class TrainingArguments:
    def __init__(
        self,
        output_dir: str,
        overwrite_output_dir: bool = False,
        cache_dir: str = None,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        data_loader_num_workers: int = 0,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-5,
        lr_scheduler_type: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ] = "cosine",
        lr_warmup_steps: int = 500,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-4,
        adam_epsilon: float = 1e-8,
        use_ema: bool = True,
        ema_inv_gamma: float = 1.0,
        ema_power: float = 3 / 4,
        ema_max_decay: float = 0.9999,
        logger: Literal["tensorboard", "wandb"] = "wandb",
        logging_dir: str = "logs",
        local_rank: int = -1,
        mixed_precision: Literal["no", "fp16", "bf16"] = "no",
        checkpointing_steps: int = 500,
        resume_from_checkpoint: str = None,
        tracker_init_kwargs: dict = {},
    ):
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.cache_dir = cache_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_loader_num_workers = data_loader_num_workers
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_warmup_steps = lr_warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.use_ema = use_ema
        self.ema_inv_gamma = ema_inv_gamma
        self.ema_power = ema_power
        self.ema_max_decay = ema_max_decay
        self.logger = logger
        self.logging_dir = logging_dir
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != local_rank:
            self.local_rank = env_local_rank
        else:
            self.local_rank = local_rank
        self.mixed_precision = mixed_precision
        self.checkpointing_steps = checkpointing_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.tracker_init_kwargs = tracker_init_kwargs


class Trainer:
    def __init__(
        self,
        project_name: str,
        wrapper: TrainingWrapper_,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        log_config: Dict = {},
    ):
        self.training_args = training_args
        self.global_step = 0

        logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            mixed_precision=training_args.mixed_precision,
            log_with=training_args.logger,
            # logging_dir=logging_dir,
        )

        if self.accelerator.is_local_main_process:
            diffusers.utils.logging.set_verbosity_info()
        else:
            diffusers.utils.logging.set_verbosity_error()

        self.accelerator.register_save_state_pre_hook(wrapper.save_model_hook)
        self.accelerator.register_load_state_pre_hook(wrapper.load_model_hook)

        self.wrapper = wrapper
        self.wrapper.register_trainer(self)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.train_batch_size,
            num_workers=training_args.data_loader_num_workers,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.eval_batch_size,
            num_workers=training_args.data_loader_num_workers,
            shuffle=False,
        )
        self.optimizers = self.wrapper.get_optimizers()
        self.schedulers: List[LRScheduler] = [
            get_scheduler(
                self.training_args.lr_scheduler_type,
                optimizer,
                num_warmup_steps=self.training_args.lr_warmup_steps
                * self.training_args.gradient_accumulation_steps,
                num_training_steps=(
                    len(self.train_dataloader) * self.training_args.num_epochs
                ),
            )
            for optimizer in self.optimizers
        ]

        # Prepare with Accelerator
        self.wrapper = self.accelerator.prepare_model(self.wrapper)
        for i in range(len(self.optimizers)):
            self.optimizers[i] = self.accelerator.prepare_optimizer(self.optimizers[i])
        for i in range(len(self.schedulers)):
            self.schedulers[i] = self.accelerator.prepare_scheduler(self.schedulers[i])
        self.train_dataloader = self.accelerator.prepare_data_loader(
            self.train_dataloader
        )
        self.val_dataloader = self.accelerator.prepare_data_loader(self.val_dataloader)

        if self.accelerator.is_main_process:
            # Got bug `first argument must be callable or None` when passing config
            self.accelerator.init_trackers(
                project_name,
                init_kwargs=self.training_args.tracker_init_kwargs,
            )
            # self.accelerator.init_trackers(project_name)

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
            if self.training_args.resume_from_checkpoint != "lastest":
                path = os.path.basename(self.training_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.training_args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.training_args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(
                    os.path.join(self.training_args.output_dir, path)
                )
                self.global_step = int(path.split("-")[1])

                resume_global_step = (
                    self.global_step * self.training_args.gradient_accumulation_steps
                )
                first_epoch = self.global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch
                    * self.training_args.gradient_accumulation_steps
                )

        # Train!
        self.wrapper.on_start()
        for epoch in range(first_epoch, self.training_args.num_epochs):
            with tqdm(
                total=num_update_steps_per_epoch,
                disable=not self.accelerator.is_local_main_process,
            ) as progress_bar:
                self.wrapper.register_progress_bar(progress_bar)
                progress_bar.set_description(f"Epoch {epoch}")

                self.wrapper.train()
                self.wrapper.on_train_epoch_start()
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

                    self.wrapper.on_train_batch_start()

                    with self.accelerator.accumulate(self.wrapper):
                        self.wrapper.training_step(batch, self.optimizers, step)
                        for scheduler in self.schedulers:
                            scheduler.step()

                    if self.accelerator.sync_gradients:
                        self.wrapper.on_train_batch_end()
                        progress_bar.update(1)

                        self.global_step += 1

                        if (
                            self.global_step % self.training_args.checkpointing_steps
                            == 0
                        ):
                            if self.accelerator.is_main_process:
                                save_path = os.path.join(
                                    self.training_args.output_dir,
                                    f"checkpoint-{self.global_step}",
                                )
                                self.accelerator.save_state(save_path)
                                logger.info(f"Saved state to {save_path}")

                    if self.accelerator.is_main_process:
                        self.wrapper.on_train_epoch_end()

            self.accelerator.wait_for_everyone()

            with tqdm(
                total=len(self.val_dataloader),
                disable=not self.accelerator.is_local_main_process,
            ) as progress_bar:
                self.wrapper.register_progress_bar(progress_bar)
                progress_bar.set_description(f"Epoch {epoch}")

                self.wrapper.eval()
                with torch.inference_mode():
                    self.wrapper.on_validation_epoch_start()
                    for step, batch in enumerate(self.val_dataloader):
                        self.wrapper.validation_step(batch, step)
                        progress_bar.update(1)

                    if self.accelerator.is_main_process:
                        self.wrapper.on_validation_epoch_end()

        self.accelerator.end_training()

    def get_tracker(self, unwrap: bool = False):
        return self.accelerator.get_tracker(self.training_args.logger, unwrap)


class TrainingWrapper(torch.nn.Module):
    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def get_optimizers(self) -> List[torch.optim.Optimizer]:
        raise NotImplementedError

    def backward_loss(self, loss: torch.Tensor):
        self.trainer.accelerator.backward(loss)

    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_train_batch_start(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @property
    def progress_bar(self):
        return self._progess_bar

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def device(self):
        return self.trainer.accelerator.device

    def log(self, values: dict, logger: bool = True, progess_bar: bool = True):
        if progess_bar:
            self.progress_bar.set_postfix(values)
        if logger and self.trainer.accelerator.is_main_process:
            self.trainer.accelerator.log(values, self.global_step)

    def register_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def register_progress_bar(self, progress_bar: tqdm):
        self._progess_bar = progress_bar

    def save_model_hook(self, models, weights, output_dir):
        raise NotImplementedError

    def load_model_hook(self, models, input_dir):
        raise NotImplementedError
