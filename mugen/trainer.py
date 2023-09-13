from typing import Dict
import torch

class TrainerWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
    
    def get_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def get_loss(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def get_metrics(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class Trainer:
    def __init__(
        self,
        model,
    ):
        pass

    def train(self)