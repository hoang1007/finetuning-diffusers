from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Iterable
from tqdm import tqdm

import torch

if TYPE_CHECKING:
    from mugen import Trainer
    from torch.optim import Optimizer


class TrainingModule(torch.nn.Module):
    def training_step(self, batch, optimizers: List[Optimizer], batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        raise NotImplementedError

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        raise NotImplementedError

    def backward_loss(self, loss: torch.Tensor):
        self.trainer.accelerator.backward(loss)

    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_train_batch_start(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @property
    def progress_bar(self):
        return self._progess_bar

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def device(self):
        return self.trainer.accelerator.device

    def log(self, values: dict, logger: bool = True, progess_bar: bool = True):
        if self.trainer.accelerator.is_main_process:
            if progess_bar:
                self.progress_bar.set_postfix(values)
            if logger and self.trainer.accelerator.is_main_process:
                self.trainer.accelerator.log(values, step=self.global_step)

    def register_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def register_progress_bar(self, progress_bar: tqdm):
        self._progess_bar = progress_bar

    def save_model_hook(self, models, weights, output_dir):
        raise NotImplementedError

    def load_model_hook(self, models, input_dir):
        raise NotImplementedError
