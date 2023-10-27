from __future__ import annotations
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
    def trainer(self) -> Trainer:
        """
        The trainer object.
        """
        return self._trainer

    @property
    def progress_bar(self):
        """
        Progress bar for the current epoch.
        """
        return self._progess_bar

    @property
    def global_step(self):
        """
        The current global step.
        """
        return self.trainer.global_step

    @property
    def device(self):
        return self.trainer.accelerator.device

    def log(self, values: dict, logger: bool = True, progess_bar: bool = True):
        """
        Log metrics for the current step to the logger and progess bar.

        Args:
            values: Dictionary of metrics to log.
            logger: Whether to log to the logger.
            progess_bar: Whether to log to the progess bar.
        """
        if self.trainer.accelerator.is_main_process:
            if progess_bar:
                self.progress_bar.set_postfix(values)
            if logger and self.trainer.accelerator.is_main_process:
                self.trainer.accelerator.log(values, step=self.global_step)

    def register_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def register_progress_bar(self, progress_bar: tqdm):
        self._progess_bar = progress_bar

    def save_pretrained(self, output_dir: str):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded.
        """
