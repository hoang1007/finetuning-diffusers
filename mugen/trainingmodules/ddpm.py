from typing import Iterable, List, Optional
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base import TrainingModule
from omegaconf import DictConfig

from diffusers.models import UNet2DModel
from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.training_utils import EMAModel


class DDPMTrainingModule(TrainingModule):
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

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x = batch[self.input_key]
        cond = batch.get(self.conditional_key, None)
        noise = torch.randn_like(x)
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
        self.log({"train/loss": loss.item()})

        return loss

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

        if self.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())
        pipeline = DDIMPipeline(self.unet, self.noise_scheduler)
        generator = torch.Generator(device=self.device).manual_seed(0)

        images = pipeline(
            batch_size=x.size(0),
            generator=generator,
            output_type='numpy'
        ).images

        if self.use_ema:
            self.ema.restore(self.unet.parameters())

        org_imgs = (x.detach() / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        self.log_images(
            {
                "original": org_imgs,
                "generated": images,
            }
        )

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        return [self.unet.parameters()]

    def save_pretrained(self, output_dir: str):
        if self.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        DDIMPipeline(self.unet, self.noise_scheduler).save_pretrained(output_dir)

        if self.use_ema:
            self.ema.restore(self.unet.parameters())
