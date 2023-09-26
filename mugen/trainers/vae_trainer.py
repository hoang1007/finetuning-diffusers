from typing import List, Optional
import os.path as osp

import torch
from torch.optim import Optimizer

from .base_trainer import TrainingWrapper
from mugen.losses import LPIPSWithDiscriminator
from omegaconf import DictConfig

from diffusers.models import AutoencoderKL
from diffusers.training_utils import EMAModel


class FreezeGradient:
    def __init__(self, module: torch.nn.Module):
        self.module = module

    def __enter__(self):
        self.state = self.module.training
        self.module.train(False)
        self.module.requires_grad_(False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.requires_grad_(True)
        self.module.train(self.state)


class VAETrainingWrapper(TrainingWrapper):
    def __init__(
        self,
        vae_config: Optional[DictConfig] = None,
        pretrained_name_or_path: Optional[str] = None,
        freeze_encoder: bool = False,
        input_key: str = "image",
        lpips_config: DictConfig = None,
        use_ema: bool = True,
    ):
        super().__init__()

        self.input_key = input_key
        self.use_ema = use_ema
        self.freeze_encoder = freeze_encoder

        assert (
            vae_config is not None or pretrained_name_or_path is not None
        ), "Either vae_config or pretrained_name_or_path must be specified!"

        if pretrained_name_or_path is not None:
            self.vae = AutoencoderKL.from_pretrained(pretrained_name_or_path)
        else:
            self.vae = AutoencoderKL.from_config(vae_config)

        if self.freeze_encoder:
            self.vae.encoder.requires_grad_(False)

        self.loss_fn = LPIPSWithDiscriminator(**lpips_config)

        if self.use_ema:
            self.ema = EMAModel(
                self.vae.parameters(),
                use_ema_warmup=True,
                model_cls=AutoencoderKL,
                model_config=self.vae.config,
            )

    def on_start(self):
        if self.use_ema:
            self.ema.to(self.device)

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
        x = batch[self.input_key]
        latent_dist = self.vae.encode(x).latent_dist
        x_recon = self.vae.decode(latent_dist.sample()).sample

        vae_opt, disc_opt = optimizers
        # Train VAE
        with FreezeGradient(self.loss_fn.discriminator):
            ae_loss, log_dict_ae = self.loss_fn(
                x,
                x_recon,
                latent_dist,
                0,
                self.global_step,
                self.get_last_layer(),
                split="train",
            )
            self.backward_loss(ae_loss)
            vae_opt.step()
            vae_opt.zero_grad()

        # Train discriminator
        with FreezeGradient(self.vae):
            disc_loss, log_dict_disc = self.loss_fn(
                x,
                x_recon,
                latent_dist,
                1,
                self.global_step,
                self.get_last_layer(),
                split="train",
            )
            self.backward_loss(disc_loss)
            disc_opt.step()
            disc_opt.zero_grad()

        log_dict = {**log_dict_ae, **log_dict_disc}
        self.log(log_dict, progess_bar=False, logger=True)
        self.log({"ae_loss": ae_loss.item()}, progess_bar=True, logger=False)

    def on_train_batch_end(self):
        if self.use_ema:
            self.ema.step(self.vae.parameters())

    def on_validation_epoch_start(self):
        self.random_batch_idx = torch.randint(
            0, len(self.trainer.val_dataloader), (1,)
        ).item()

    def validation_step(self, batch, batch_idx: int):
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return

        x = batch[self.input_key]
        latent_dist = self.vae.encode(x).latent_dist
        z = latent_dist.sample()
        x_recon = self.vae.decode(z).sample
        x_gen = self.vae.decode(torch.randn_like(z)).sample

        to_np_images = lambda x: (x.detach() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()
        self.trainer.get_tracker().log_images(
            {
                "original": to_np_images(x),
                "reconstruction": to_np_images(x_recon),
                "generated": to_np_images(x_gen),
            }
        )

    def get_optimizers(self) -> List[Optimizer]:
        vae_opt = torch.optim.AdamW(
            self.vae.decoder.parameters()
            if self.freeze_encoder
            else self.vae.parameters(),
            lr=self.trainer.training_args.learning_rate,
            weight_decay=self.trainer.training_args.adam_weight_decay,
            betas=(
                self.trainer.training_args.adam_beta1,
                self.trainer.training_args.adam_beta2,
            ),
            eps=self.trainer.training_args.adam_epsilon,
        )

        disc_opt = torch.optim.AdamW(
            self.loss_fn.discriminator.parameters(),
            lr=self.trainer.training_args.learning_rate,
            weight_decay=self.trainer.training_args.adam_weight_decay,
            betas=(
                self.trainer.training_args.adam_beta1,
                self.trainer.training_args.adam_beta2,
            ),
            eps=self.trainer.training_args.adam_epsilon,
        )

        return [vae_opt, disc_opt]

    def save_model_hook(self, models, weights, output_dir):
        if self.use_ema:
            self.ema.save_pretrained(osp.join(output_dir, "vae_ema"))

        for i, model in enumerate(models):
            model.vae.save_pretrained(osp.join(output_dir, f"vae"))
            weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.use_ema:
            load_model = EMAModel.from_pretrained(
                osp.join(input_dir, "vae_ema"), AutoencoderKL
            )
            self.ema.load_state_dict(load_model.state_dict())
            self.ema.to(self.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = AutoencoderKL.from_pretrained(input_dir, subfolder="vae")
            model.vae.register_to_config(**load_model.config)

            model.vae.load_state_dict(load_model.state_dict())
            del load_model
