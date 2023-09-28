from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self):
        self._correct = 0
        self._total = 0

    def update(self, pred, target):
        self._correct += (pred == target).sum().item()
        self._total += target.numel()

    def compute(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self):
        self._correct = 0
        self._total = 0
