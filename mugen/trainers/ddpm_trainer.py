from typing import List, Optional
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base_trainer import TrainingWrapper
from mugen.losses import LPIPSWithDiscriminator
from omegaconf import DictConfig

from diffusers.models import UNet2DModel
from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.training_utils import EMAModel

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


class DDPMTrainingWrapper(TrainingWrapper):
    def __init__(
        self,
        unet_config: Optional[DictConfig] = None,
        pretrained_name_or_path: Optional[str] = None,
        scheduler_config: Optional[DictConfig] = None,
        input_key: str = "image",
        conditional_key: str = "label",
        use_ema: bool = True,
        enable_xformers_memory_efficient_attention: bool = False
    ):
        super().__init__()

        self.input_key = input_key
        self.conditional_key = conditional_key
        self.use_ema = use_ema

        assert (
            unet_config is not None or pretrained_name_or_path is not None
        ), "Either unet_config or pretrained_name_or_path must be specified!"
        if pretrained_name_or_path is not None:
            try:
                self.unet = UNet2DModel.from_pretrained(pretrained_name_or_path, subfolder="unet")
                self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_name_or_path, subfolder="scheduler")
            except:
                self.unet = UNet2DModel.from_pretrained(pretrained_name_or_path)
                self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_name_or_path)
        else:
            assert scheduler_config is not None, "scheduler_config must be specified if unet_config is specified!"
            self.unet = UNet2DModel(**unet_config)
            self.noise_scheduler = DDIMScheduler(**scheduler_config)

        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()

        if self.use_ema:
            self.ema = EMAModel(
                self.unet.parameters(),
                use_ema_warmup=True,
                model_cls=UNet2DModel,
                model_config=self.unet.config,
            )
    
    def on_start(self):
        if self.use_ema:
            self.ema.to(self.device)

    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
        x = batch[self.input_key]
        cond = batch.get(self.conditional_key, None)
        noise = torch.rand_like(x)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x.size(0),),
            device=x.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        unet_output = self.unet(noisy_x, timesteps, class_labels=cond).sample

        # predict the noise
        loss = F.mse_loss(unet_output, noise)
        self.backward_loss(loss)

        if self.trainer.accelerator.sync_gradients:
            self.trainer.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
        opt = optimizers[0]
        opt.step()
        opt.zero_grad()

        self.log({"train/loss": loss.item()})

    def on_train_batch_end(self):
        if self.use_ema:
            self.ema.step(self.unet.parameters())
    
    def on_validation_epoch_start(self):
        self.random_batch_idx = torch.randint(
            0, len(self.trainer.val_dataloader), (1,)
        ).item()

    def validation_step(self, batch, batch_idx: int):
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return
        
        x = batch[self.input_key]
        cond = batch.get(self.conditional_key, None)

        pipeline = DDIMPipeline(self.unet, self.noise_scheduler)
        generator = torch.Generator(device=self.device).manual_seed(0)

        images = pipeline(
            batch_size=x.size(0),
            generator=generator,
            output_type='numpy'
        ).images

        org_imgs = (x.detach() / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        self.trainer.get_tracker().log_images(
            {
                "original": org_imgs,
                "generation": images,
            }
        )

    def get_optimizers(self) -> List[Optimizer]:
        opt = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.trainer.training_args.learning_rate,
            weight_decay=self.trainer.training_args.adam_weight_decay,
            betas=(
                self.trainer.training_args.adam_beta1,
                self.trainer.training_args.adam_beta2,
            ),
            eps=self.trainer.training_args.adam_epsilon,
        )

        return [opt]

    def save_model_hook(self, models, weights, output_dir):
        if self.use_ema:
            self.ema.save_pretrained(osp.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.unet.save_pretrained(osp.join(output_dir, "unet"))
            weights.pop()
        
        self.noise_scheduler.save_pretrained(osp.join(output_dir, "scheduler"))

    def load_model_hook(self, models, input_dir):
        if self.use_ema:
            load_model = EMAModel.from_pretrained(
                osp.join(input_dir, "unet_ema"), UNet2DModel
            )
            self.ema.load_state_dict(load_model.state_dict())
            self.ema.to(self.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
            model.unet.register_to_config(**load_model.config)

            model.unet.load_state_dict(load_model.state_dict())
            del load_model
        
        self.noise_scheduler.from_pretrained(input_dir, subfolder="scheduler")
