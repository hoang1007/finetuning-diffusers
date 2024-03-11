from typing import List, Optional, Iterable, Literal
import os

import torch
import torch.nn.functional as F

from lightning_accelerate import TrainingModule
from lightning_accelerate.metrics import MeanMetric

from diffusers.models import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel

from mugen.loaders import load_pipeline


class Text2ImageTrainingModule(TrainingModule):
    UNET_TARGET_MODULES = [
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
    CLIP_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]
    LORA_TARGET_MODULES = UNET_TARGET_MODULES + CLIP_TARGET_MODULES

    def __init__(
        self,
        pretrained_name_or_path: str,
        tokenizer_pretrained_name_or_path: Optional[str] = None,
        vae_pretrained_name_or_path: Optional[str] = None,
        text_encoder_pretrained_name_or_path: Optional[str] = None,
        scheduler_pretrained_name_or_path: Optional[str] = None,
        safety_checker_pretrained_name_or_path: Optional[str] = None,
        feature_extractor_pretrained_name_or_path: Optional[str] = None,
        is_from_original_sd: bool = False,
        train_text_encoder: bool = True,
        clip_skip: int = 0,
        use_latent_input: bool = False,
        snr_gamma: Optional[float] = None,
        prediction_type: Optional[Literal["epsilon", "v_prediction"]] = None,
        use_ema: bool = True,
        enable_xformers_memory_efficient_attention: bool = False,
        enable_gradient_checkpointing: bool = False,
        run_safety_checker: bool = True,
    ):
        """Training Module for Text2Image

        Args:
            pretrained_name_or_path (Optional[str], optional): Pretrained path of base model.
            It could be a model's name from hub, local checkpoint directory, single checkpoint file with diffusers format or original format.
            tokenizer_pretrained_name_or_path (Optional[str], optional): Tokenizer path to override the base. Defaults to None.
            vae_pretrained_name_or_path (Optional[str], optional): VAE pretrained path to override the base. Defaults to None.
            text_encoder_pretrained_name_or_path (Optional[str], optional): Text Encoder pretrained path to override the base. Defaults to None.
            scheduler_pretrained_name_or_path (Optional[str], optional): Scheduler path to override the base. Defaults to None.
            safety_checker_pretrained_name_or_path (Optional[str], optional): Safety checker to override the base. Defaults to None.
            feature_extractor_pretrained_name_or_path (Optional[str], optional): Pretrained path of feature extractor of safety checker to override the base. Defaults to None.
            is_from_original_sd (bool, optional): Whether provided `pretrained_name_or_path` is original (CompVis) format or not. Defaults to False.
            train_text_encoder (bool, optional): Whether to train text encoder or not. Defaults to True.
            clip_skip (int, optional): The number of layers of CLIP text encoder to skip. Higher number can lead to better quality. Defaults to 0.
            use_latent_input (bool, optional): If `True`, model will assume the provided input is latent. Defaults to False.
            snr_gamma (Optional[float], optional): Min snr gamma. Set to 5.0 is recommended. Defaults to None.
            prediction_type (Optional[Literal[&quot;epsilon&quot;, &quot;v_prediction&quot;]], optional): Prediction type. Defaults to None (will be infer from scheduler)
            use_ema (bool, optional): Use EMA. Defaults to True.
            enable_xformers_memory_efficient_attention (bool, optional): Enable xformers. Defaults to False.
            enable_gradient_checkpointing (bool, optional): Enable gradient checkpoint. This will slightly reduce memory consuming. Defaults to False.
            run_safety_checker (bool, optional): Whether to run safety checker to detect NFSW content or not. Defaults to True.
        """
        super().__init__()

        self.input_key = "image"
        self.caption_key = "text"

        pipeline = load_pipeline(pretrained_name_or_path, is_from_original_sd)

        self.unet = pipeline.unet

        if scheduler_pretrained_name_or_path is not None:
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                scheduler_pretrained_name_or_path
            )
        else:
            self.noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        if text_encoder_pretrained_name_or_path is not None:
            self.text_encoder = CLIPTextModel.from_pretrained(
                text_encoder_pretrained_name_or_path
            )
        else:
            self.text_encoder = pipeline.text_encoder

        if clip_skip > 0:
            self.text_encoder.text_model.encoder.layers = (
                self.text_encoder.text_model.encoder.layers[:-clip_skip]
            )

        if tokenizer_pretrained_name_or_path is not None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_pretrained_name_or_path
            )
        else:
            self.tokenizer = pipeline.tokenizer

        if not use_latent_input:
            if vae_pretrained_name_or_path is not None:
                self.vae = AutoencoderKL.from_pretrained(vae_pretrained_name_or_path)
            else:
                self.vae = pipeline.vae
        else:
            self.input_key = "latent"

        if not train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if prediction_type is not None:
            self.noise_scheduler.register_to_config(prediction_type=prediction_type)

        if self.config.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()

        if enable_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

        if self.config.use_ema:
            self.ema = EMAModel(
                self.unet.parameters(),
                use_ema_warmup=True,
                model_cls=UNet2DConditionModel,
                model_config=self.unet.config,
            )

        self.vae_config = pipeline.vae.config
        self.loss_log = MeanMetric()

    def on_start(self):
        if self.config.use_ema:
            self.ema.to(self.device)

    def get_latents(self, batch):
        if self.config.use_latent_input:
            latents = batch[self.input_key]
        else:
            imgs = batch[self.input_key]
            latents = self.vae.encode(imgs).latent_dist.sample()

        return latents

    def get_text_embeds(self, batch):
        texts = batch[self.caption_key]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        for k in encoded:
            encoded[k] = encoded[k].to(self.device)
        text_embeds = self.text_encoder(**encoded).last_hidden_state
        return text_embeds

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        latents = self.get_latents(batch) * self.vae_config.scaling_factor
        encoder_hidden_states = self.get_text_embeds(batch)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.size(0),),
            device=latents.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(latents, noise, timesteps)
        unet_output = self.unet(noisy_x, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # predict the noise
        if self.config.snr_gamma is None:
            loss = F.mse_loss(unet_output, target, reduction="mean")
        else:
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = (
                torch.stack(
                    [snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(unet_output, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        self.loss_log.update(loss.item())
        self.log({"train/loss": self.loss_log.compute()})

        return loss

    def on_train_batch_end(self):
        if self.config.use_ema:
            self.ema.step(self.unet.parameters())

    def on_train_epoch_end(self):
        self.loss_log.reset()

    def on_validation_epoch_start(self):
        self.random_batch_idx = torch.randint(
            0, len(self.trainer.val_dataloader), (1,)
        ).item()

    def validation_step(self, batch, batch_idx: int):
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return

        encoder_hidden_states = self.get_text_embeds(batch)

        if self.config.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        pipeline = self.get_pipeline().to(self.device)

        if self.config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        real_images = pipeline.vae.decode(self.get_latents(batch), return_dict=False)[0]
        real_images = (real_images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        real_images = real_images.cpu().permute(0, 2, 3, 1).float().numpy()
        gen_images = pipeline(
            prompt_embeds=encoder_hidden_states,
            output_type="np",
        ).images
        del pipeline

        if self.config.use_ema:
            self.ema.restore(self.unet.parameters())

        self.log_images({"generated": gen_images, "real": real_images})

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        params = [{"params": self.unet.parameters()}]
        if self.config.train_text_encoder:
            params.append({"params": self.text_encoder.parameters(), "lr_scale": 0.5})
        return params

    def save_pretrained(self, output_dir: str):
        if self.config.use_ema:
            self.ema.store(self.unet.parameters())
            self.ema.copy_to(self.unet.parameters())

        self.get_pipeline().save_pretrained(output_dir)

        if self.config.use_ema:
            self.ema.restore(self.unet.parameters())

    def get_pipeline(self):
        if (
            self.config.use_latent_input
            and self.config.vae_pretrained_name_or_path is not None
        ):
            vae = AutoencoderKL.from_pretrained(self.config.vae_pretrained_name_or_path)
        elif not self.config.use_latent_input:
            vae = self.vae
        else:
            vae = None

        return load_pipeline(
            pretrained_name_or_path=self.config.pretrained_name_or_path,
            is_from_original_sd=self.config.is_from_original_sd,
            vae=vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr
