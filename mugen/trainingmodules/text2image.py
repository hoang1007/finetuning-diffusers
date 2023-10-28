from typing import List, Optional, Iterable
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from .base import TrainingModule
from omegaconf import DictConfig

from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers import StableDiffusionPipeline, DDPMScheduler, PNDMScheduler
from diffusers.training_utils import EMAModel


class Text2ImageTrainingModule(TrainingModule):
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
        tokenizer_pretrained_name_or_path: Optional[str] = None,
        vae_pretrained_name_or_path: Optional[str] = None,
        text_encoder_pretrained_name_or_path: Optional[str] = None,
        scheduler_pretrained_name_or_path: Optional[str] = None,
        safety_checker_pretrained_name_or_path: Optional[str] = None,
        feature_extractor_pretrained_name_or_path: Optional[str] = None,
        input_key: str = "latent",
        conditional_key: str = "text_embedding",
        use_ema: bool = True,
        enable_xformers_memory_efficient_attention: bool = False,
        enable_gradient_checkpointing: bool = False,
        run_safety_checker: bool = True,
    ):
        super().__init__()

        self.input_key = input_key
        self.conditional_key = conditional_key
        self.use_ema = use_ema
        self.pretrained_name_or_path = pretrained_name_or_path
        self.vae_pretrained_name_or_path = vae_pretrained_name_or_path
        self.tokenizer_pretrained_name_or_path = tokenizer_pretrained_name_or_path
        self.text_encoder_pretrained_name_or_path = text_encoder_pretrained_name_or_path
        self.safety_checker_pretrained_name_or_path = safety_checker_pretrained_name_or_path
        self.feature_extractor_pretrained_name_or_path = feature_extractor_pretrained_name_or_path
        self.scheduler_pretrained_name_or_path = scheduler_pretrained_name_or_path
        self.run_safety_checker = run_safety_checker

        self.enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )

        if unet_config is not None:
            self.unet = UNet2DConditionModel(**unet_config)
        elif unet_pretrained_name_or_path is not None:
            self.unet = UNet2DConditionModel.from_pretrained(unet_pretrained_name_or_path)
        elif pretrained_name_or_path is not None:
            self.unet = UNet2DConditionModel.from_pretrained(pretrained_name_or_path, subfolder="unet")
        else:
            raise ValueError("Either `unet_config` or `pretrained_name_or_path` or `unet_pretrained_name_or_path` must be specified!")

        if scheduler_pretrained_name_or_path is not None:
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
        
        if enable_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if self.use_ema:
            self.ema = EMAModel(
                self.unet.parameters(),
                use_ema_warmup=True,
                model_cls=UNet2DConditionModel,
                model_config=self.unet.config,
            )
    
        self.vae_config = self.get_pipeline().vae.config

    def on_start(self):
        if self.use_ema:
            self.ema.to(self.device)

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x = batch[self.input_key] * self.vae_config.scaling_factor
        cond = batch[self.conditional_key]
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x.size(0),),
            device=x.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        unet_output = self.unet(noisy_x, timesteps, cond).sample

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

        cond = batch[self.conditional_key]

        if self.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        pipeline = self.get_pipeline().to(self.device)

        if self.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        real_images = pipeline.vae.decode(batch[self.input_key], return_dict=False)[0]
        real_images = (real_images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        real_images = real_images.cpu().permute(0, 2, 3, 1).float().numpy()
        gen_images = pipeline(prompt_embeds=cond, output_type="np").images
        del pipeline

        if self.use_ema:
            self.ema.restore(self.unet.parameters())

        self.log_images({"generated": gen_images, "real": real_images})

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        return [self.unet.parameters()]

    def save_pretrained(self, output_dir: str):
        if self.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        self.get_pipeline().save_pretrained(output_dir)

        if self.use_ema:
            self.ema.restore(self.unet.parameters())

    def get_pipeline(self):
        if self.vae_pretrained_name_or_path:
            vae = AutoencoderKL.from_pretrained(self.vae_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            vae = AutoencoderKL.from_pretrained(self.pretrained_name_or_path, subfolder="vae")
        else:
            raise ValueError("Either `vae_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")
        
        if self.text_encoder_pretrained_name_or_path:
            text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            text_encoder = CLIPTextModel.from_pretrained(self.pretrained_name_or_path, subfolder="text_encoder")
        else:
            raise ValueError("Either `text_encoder_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")
        
        if self.tokenizer_pretrained_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_name_or_path, subfolder="tokenizer")
        else:
            raise ValueError("Either `tokenizer_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")

        if not self.run_safety_checker:
            safety_checker = None
        elif self.safety_checker_pretrained_name_or_path:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(self.safety_checker_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(self.pretrained_name_or_path, subfolder="safety_checker")
        else:
            raise ValueError("Either `safety_checker_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")

        if self.feature_extractor_pretrained_name_or_path:
            feature_extractor = CLIPImageProcessor.from_pretrained(self.feature_extractor_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            feature_extractor = CLIPImageProcessor.from_pretrained(self.pretrained_name_or_path, subfolder="feature_extractor")
        else:
            raise ValueError("Either `feature_extractor_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")
        
        if self.scheduler_pretrained_name_or_path:
            scheduler = PNDMScheduler.from_pretrained(self.scheduler_pretrained_name_or_path)
        elif self.pretrained_name_or_path:
            scheduler = PNDMScheduler.from_pretrained(self.pretrained_name_or_path, subfolder="scheduler")
        else:
            raise ValueError("Either `scheduler_pretrained_name_or_path` or `pretrained_name_or_path` must be specified!")

        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=self.unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=self.run_safety_checker,
        )

        return pipeline
