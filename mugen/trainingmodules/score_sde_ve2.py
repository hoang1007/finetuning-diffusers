from typing import Union, Optional, Dict
import math
import torch

from lightning_accelerate.utils.config_utils import ConfigMixin
from lightning_accelerate import TrainingModule
from mugen.models.bruh import DenoiseEstimator
from diffusers import UNet1DModel
from tqdm import tqdm


class ScoreSdeVeTrainingModule(TrainingModule):
    def __init__(
        self, unet_config: Dict, scheduler_config: Dict, input_key: str = "image"
    ):
        super().__init__()

        self.unet = DenoiseEstimator(**unet_config)
        self.scheduler = ScoreSdeVeScheduler(**scheduler_config)

    def training_step(self, batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        samples = batch[self.config.input_key]
        batch_size, n_channels, h, w = samples.shape
        # import pdb; pdb.set_trace()
        device = samples.device

        # timesteps = torch.randint(
        #     0,
        #     self.scheduler.num_train_timesteps,
        #     (batch_size,),
        #     device=device,
        #     dtype=torch.long,
        # )
        timesteps = torch.rand(batch_size, device=device) * (1 - self.scheduler.sampling_eps) + self.scheduler.sampling_eps
        perturbed, sigmas, noise = self.scheduler.add_noise(samples, timesteps)
        score = self.unet(perturbed, timesteps)

        # Likelihood weighting loss
        # diffusion = (
        #     sigmas
        #     * torch.tensor(
        #         2
        #         * (
        #             math.log(self.scheduler.sigma_max)
        #             - math.log(self.scheduler.sigma_min)
        #         ),
        #         device=device,
        #     ).sqrt()
        # )

        # losses = torch.square(score + noise / sigmas[:, None, None, None])
        # loss = losses.view(batch_size, -1).mean(-1) * diffusion
        losses = torch.square(score * sigmas[:, None, None, None] + noise)
        loss = losses.flatten(start_dim=1).mean(-1)
        loss = loss.mean()

        self.log({"train/loss": loss.item()})

        return loss

    def on_validation_epoch_start(self):
        self.random_batch_idx = torch.randint(
            0, len(self.trainer.val_dataloader), (1,)
        ).item()

    def validation_step(self, batch, batch_idx: int):
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return

        x = batch[self.config.input_key]
        images = self.sample(
            len(x), self.scheduler.num_train_timesteps, output_type="numpy"
        )

        org_imgs = (x.detach() / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        self.log_images(
            {
                "original": org_imgs,
                "generated": images,
            }
        )

    def get_optim_params(self):
        return [self.unet.parameters()]

    def save_pretrained(self, output_dir: str):
        # self.unet.save_pretrained(output_dir, safe_serialization=False)
        pass

    def sample(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
    ):
        device = self.device
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        n_channels, h, w = 1, 32, 32
        samples = (
            torch.randn(
                (
                    batch_size,
                    n_channels, h, w
                ),
                generator=generator,
                device=device,
            )
            * self.scheduler.init_noise_sigma
        )

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # sigma_t = self.scheduler.sigmas[i] * torch.ones(batch_size, device=device)
            ts = t * torch.ones(batch_size, device=device)
            for _ in range(self.scheduler.correct_steps):
                # model_out = self.unet(samples, sigma_t).sample
                model_out = self.unet(samples, ts)
                samples = self.scheduler.step_correct(model_out, samples, generator)

            # model_out = self.unet(samples, sigma_t).sample
            model_out = self.unet(samples, ts)
            samples, samples_mean = self.scheduler.step_pred(
                model_out, t, samples, generator
            )
        print(samples_mean.mean(), samples_mean.std())
        samples = samples_mean.clamp(-1, 1)
        # samples = samples_mean
        if output_type == "numpy":
            samples = samples.cpu().permute(0, 2, 3, 1).numpy()

        return samples


# Adapt from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_sde_ve.py
class ScoreSdeVeScheduler(ConfigMixin):
    def __init__(
        self,
        num_train_timesteps: int = 2000,
        snr: float = 0.15,
        sigma_min: float = 0.01,
        sigma_max: float = 1348.0,
        sampling_eps: float = 1e-5,
        correct_steps: int = 1,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.snr = snr
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sampling_eps = sampling_eps
        self.correct_steps = correct_steps

        self.init_noise_sigma = sigma_max
        self.timesteps = None

        self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        sampling_eps: float = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the continuous timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, *optional*):
                The final timestep value (overrides value given during scheduler instantiation).
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

        """
        sampling_eps = sampling_eps if sampling_eps is not None else self.sampling_eps

        self.timesteps = torch.linspace(
            1, sampling_eps, num_inference_steps, device=device
        )

    def set_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: float = None,
        sigma_max: float = None,
        sampling_eps: float = None,
    ):
        """
        Sets the noise scales used for the diffusion chain (to be run before inference). The sigmas control the weight
        of the `drift` and `diffusion` components of the sample update.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                The initial noise scale value (overrides value given during scheduler instantiation).
            sigma_max (`float`, optional):
                The final noise scale value (overrides value given during scheduler instantiation).
            sampling_eps (`float`, optional):
                The final timestep value (overrides value given during scheduler instantiation).

        """
        sigma_min = sigma_min if sigma_min is not None else self.sigma_min
        sigma_max = sigma_max if sigma_max is not None else self.sigma_max
        sampling_eps = sampling_eps if sampling_eps is not None else self.sampling_eps
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        # self.sigmas = sigma_min * (sigma_max / sigma_min) ** (
        #     self.timesteps / sampling_eps
        # )
        self.discrete_sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_min), math.log(sigma_max), num_inference_steps
            )
        )
        self.sigmas = torch.tensor(
            [sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps]
        )

    def get_adjacent_sigma(self, timesteps, t):
        return torch.where(
            timesteps == 0,
            torch.zeros_like(t.to(timesteps.device)),
            self.discrete_sigmas[timesteps - 1].to(timesteps.device),
        )

    def step_pred(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`] is returned, otherwise a tuple
                is returned where the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        timestep = timestep * torch.ones(
            sample.shape[0], device=sample.device
        )  # torch.repeat_interleave(timestep, sample.shape[0])
        timesteps = (timestep * (len(self.timesteps) - 1)).long()

        # mps requires indices to be in the same device, so we use cpu as is the default with cuda
        timesteps = timesteps.to(self.discrete_sigmas.device)

        sigma = self.discrete_sigmas[timesteps].to(sample.device)
        adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample.device)
        drift = torch.zeros_like(sample)
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # equation 6 in the paper: the model_output modeled by the network is grad_x log pt(x)
        # also equation 47 shows the analog from SDE models to ancestral sampling methods
        diffusion = diffusion.flatten()
        while len(diffusion.shape) < len(sample.shape):
            diffusion = diffusion.unsqueeze(-1)
        drift = drift - diffusion**2 * model_output

        #  equation 6: sample noise for the diffusion term of
        noise = torch.randn(
            sample.shape,
            layout=sample.layout,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        prev_sample_mean = (
            sample - drift
        )  # subtract because `dt` is a small negative timestep
        # TODO is the variable diffusion the correct scaling term for the noise?
        prev_sample = (
            prev_sample_mean + diffusion * noise
        )  # add impact of diffusion field g

        return prev_sample, prev_sample_mean

    def step_correct(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Correct the predicted sample based on the `model_output` of the network. This is often run repeatedly after
        making the prediction for the previous timestep.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_sde_ve.SdeVeOutput`] is returned, otherwise a tuple
                is returned where the first element is the sample tensor.

        """
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = torch.randn(sample.shape, layout=sample.layout, generator=generator).to(
            sample.device
        )

        # compute step size from the model_output, the noise, and the snr
        grad_norm = torch.norm(
            model_output.reshape(model_output.shape[0], -1), dim=-1
        ).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # compute corrected sample: model_output term and noise term
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)
        prev_sample_mean = sample + step_size * model_output
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
    ):
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        timesteps = timesteps.to(original_samples.device)
        if timesteps.dtype == torch.long:
            sigmas = self.sigmas.to(original_samples.device)[timesteps]
        elif timesteps.dtype == torch.float:
            sigmas = self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, timesteps)
        else:
            raise ValueError

        if noise is None:
            noise = torch.randn_like(original_samples)

        noisy_samples = noise * sigmas[:, None, None, None] + original_samples
        return noisy_samples, sigmas, noise
    # def add_noise(
    #     self,
    #     original_samples: torch.FloatTensor,
    #     noise: torch.FloatTensor,
    #     timesteps: torch.FloatTensor,
    # ) -> torch.FloatTensor:
    #     # Make sure sigmas and timesteps have the same device and dtype as original_samples
    #     timesteps = timesteps.to(original_samples.device)
    #     sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
    #     noise = (
    #         noise * sigmas[:, None, None, None]
    #         if noise is not None
    #         else torch.randn_like(original_samples) * sigmas[:, None, None, None]
    #     )
    #     noisy_samples = noise + original_samples
    #     return noisy_samples

    def __len__(self):
        return self.num_train_timesteps
