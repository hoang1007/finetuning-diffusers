from typing import List, Optional, Iterable
from warnings import warn

import torch

from .base import TrainingModule
from mugen.losses.lpips import LPIPSWithDiscriminator, is_lpips_available
from omegaconf import DictConfig

from diffusers.models import AutoencoderKL
from diffusers.training_utils import EMAModel


def freeze(module: torch.nn.Module):
    module.eval()
    module.requires_grad_(False)


def unfreeze(module: torch.nn.Module):
    module.requires_grad_(True)
    module.train()


class VAETrainingModule(TrainingModule):
    def __init__(
        self,
        vae_config: Optional[DictConfig] = None,
        pretrained_name_or_path: Optional[str] = None,
        freeze_encoder: bool = False,
        input_key: str = "image",
        lpips_config: Optional[DictConfig] = None,
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

        if lpips_config is not None:
            assert is_lpips_available(), "LPIPS is not available!"
            self.loss_fn = LPIPSWithDiscriminator(**lpips_config)
            self._use_lpips = True
        else:
            self.loss_fn = torch.nn.MSELoss()
            self._use_lpips = False

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

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x = batch[self.input_key]
        latent_dist = self.vae.encode(x).latent_dist
        x_recon = self.vae.decode(latent_dist.sample()).sample

        if self._use_lpips:
            if optimizer_idx == 0:
                freeze(self.loss_fn.discriminator)
                unfreeze(self.vae.encoder)
            elif optimizer_idx == 1:
                freeze(self.vae.encoder)
                unfreeze(self.loss_fn.discriminator)

            loss, log_dict = self.loss_fn(
                x,
                x_recon,
                latent_dist,
                optimizer_idx,
                self.global_step,
                self.get_last_layer(),
                split="train",
            )
            self.log(log_dict)
            return loss
        else:
            loss = self.loss_fn(x, x_recon)
            self.log("train/loss", loss.item())
            return loss

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

        to_np_images = (
            lambda x: (x.detach() / 2 + 0.5)
            .clamp(0, 1)
            .permute(0, 2, 3, 1)
            .float()
            .cpu()
            .numpy()
        )
        self.log_images(
            {
                "original": to_np_images(x),
                "reconstruction": to_np_images(x_recon),
                "generated": to_np_images(x_gen),
            }
        )

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        if self._use_lpips:
            return [self.vae.parameters(), self.loss_fn.discriminator.parameters()]
        else:
            return [self.vae.parameters()]

    def save_pretrained(self, output_dir: str):
        if self.use_ema:
            self.ema.store(self.vae.parameters())
            self.ema.copy_to(self.vae.parameters())

        self.vae.save_pretrained(output_dir)

        if self.use_ema:
            self.ema.restore(self.vae.parameters())
