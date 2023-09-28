from typing import Union
import torch
from .base import BaseMetric


class MeanMetric(BaseMetric):
    def __init__(self):
        self._store = []
    
    def update(self, val: Union[torch.Tensor, float]):
        self._store.append(val)

    def compute(self):
        if len(self._store) == 0:
            return 0.0
        else:
            return sum(self._store) / len(self._store)
    
    def reset(self):
        self._store.clear()
