from __future__ import annotations
from warnings import warn
from typing import TYPE_CHECKING
from typing import List, Iterable
from tqdm import tqdm

import torch
from mugen.hooks import BaseHook

if TYPE_CHECKING:
    from mugen import Trainer
    from torch.optim import Optimizer


class TrainingModule(torch.nn.Module, BaseHook):
    LORA_TARGET_MODULES = None

    def training_step(self, batch, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """
        Args:
            batch: The current batch.
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the current optimizer.
        
        Returns:
            Tensor containing the loss for the current step.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        """
        Args:
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        raise NotImplementedError

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        """
        The parameters to optimize.

        Returns:
            List of parameter groups. Each parameter group in the return value will be passed to an optimizer.
        """
        raise NotImplementedError

    @property
    def trainer(self):
        """
        The trainer object.
        """
        return self._trainer if hasattr(self, "_trainer") else None

    @property
    def progress_bar(self):
        """
        Progress bar for the current epoch.
        """
        return self._progess_bar if hasattr(self, "_progess_bar") else None

    @property
    def global_step(self):
        """
        The current global step.
        """
        return self.trainer.global_step if self.trainer is not None else 0

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def log(self, values: dict, logger: bool = True, progess_bar: bool = True):
        """
        Log metrics for the current step to the logger and progess bar.

        Args:
            values: Dictionary of metrics to log.
            logger: Whether to log to the logger.
            progess_bar: Whether to log to the progess bar.
        """
        if self.trainer is None:
            warn("No trainer is registered to the training module!")
            return

        if self.trainer.accelerator.is_main_process:
            if progess_bar:
                if self.progress_bar is not None:
                    self.progress_bar.set_postfix(values)
                else:
                    self.trainer.accelerator.print("No progess bar found. Skipping progess bar logging.")
            if logger:
                self.trainer.accelerator.log(values, step=self.global_step)

    def log_images(self, images: dict):
        """
        Log images for the current step to the logger.

        Args:
            images: Dictionary of images to log.
        """
        if self.trainer is None:
            warn("No trainer is registered to the training module!")
            return

        if self.trainer.accelerator.is_main_process:
            tracker = self.trainer.get_tracker()
            if tracker is None:
                self.trainer.accelerator.print("No tracker found. Skipping image logging.")
            elif hasattr(tracker, "log_images"):
                tracker.log_images(images, step=self.global_step)
            else:
                self.trainer.accelerator.print(f"Tracker {tracker.__class__.__name__} does not support image logging.")

    def register_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def register_progress_bar(self, progress_bar: tqdm):
        self._progess_bar = progress_bar

    def save_pretrained(self, output_dir: str):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded.
        """
