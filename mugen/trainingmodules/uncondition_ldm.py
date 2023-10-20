from typing import List, Optional, Iterable
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base import TrainingModule
from omegaconf import DictConfig

from diffusers.models import UNet2DModel, AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import DDPMScheduler, PNDMScheduler, LDMPipeline
from diffusers.training_utils import EMAModel


class LDMTrainingModule(TrainingModule):
    LORA_TARGET_MODULES = [
        "to_q",
        "to_k",
        "to_v",
        "proj",
        "proj_in",
        "proj_out",
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "to_out.0",
        "time_emb_proj",
        "ff.net.2",
    ]

    def __init__(
        self,
        pretrained_name_or_path: Optional[str] = None,
        unet_config: Optional[DictConfig] = None,
        unet_pretrained_name_or_path: Optional[str] = None,
        vae_pretrained_name_or_path: Optional[str] = None,
        scheduler_config: Optional[DictConfig] = None,
        scheduler_pretrained_name_or_path: Optional[str] = None,
        input_key: str = "latent",
        use_ema: bool = True,
        enable_xformers_memory_efficient_attention: bool = False,
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = use_ema
        self.pretrained_name_or_path = pretrained_name_or_path
        self.vae_pretrained_name_or_path = vae_pretrained_name_or_path
        self.scheduler_pretrained_name_or_path = scheduler_pretrained_name_or_path

        self.enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )

        if unet_config is not None:
            self.unet = UNet2DModel(**unet_config)
        elif unet_pretrained_name_or_path is not None:
            self.unet = UNet2DModel.from_pretrained(unet_pretrained_name_or_path)
        elif pretrained_name_or_path is not None:
            self.unet = UNet2DModel.from_pretrained(pretrained_name_or_path, subfolder="unet")
        else:
            raise ValueError("Either `unet_config` or `pretrained_name_or_path` or `unet_pretrained_name_or_path` must be specified!")

        if scheduler_config is not None:
            self.noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler_pretrained_name_or_path is not None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                scheduler_pretrained_name_or_path
            )
        elif pretrained_name_or_path is not None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                pretrained_name_or_path, subfolder="scheduler"
            )
        else:
            raise ValueError("Either `scheduler_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")

        if self.enable_xformers_memory_efficient_attention:
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
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x.size(0),),
            device=x.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        unet_output = self.unet(noisy_x, timesteps).sample

        # predict the noise
        loss = F.mse_loss(unet_output, noise)
        self.backward_loss(loss)

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
        x = batch[self.input_key]
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return

        if self.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        pipeline = self.get_pipeline().to(self.device)

        if self.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        images = pipeline(batch_size=x.size(0), output_type="np").images

        if self.use_ema:
            self.ema.restore(self.unet.parameters())

        self.trainer.get_tracker().log_images({"generated": images})

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        return [self.unet.parameters()]

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

    def get_pipeline(self):
        if self.vae_pretrained_name_or_path:
            vae = AutoencoderKL.from_pretrained(self.vae_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            vae = AutoencoderKL.from_pretrained(self.pretrained_name_or_path, subfolder="vae")
        else:
            raise ValueError("Either `vae_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")

        scheduler = PNDMScheduler.from_config(self.noise_scheduler.config)

        pipeline = LDMPipeline(
            vqvae=vae,
            unet=self.unet,
            scheduler=scheduler
        )

        return pipeline
