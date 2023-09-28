from abc import abstractmethod
from numbers import Number


class BaseMetric:
    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the metric with a new value.
        """
        pass

    @abstractmethod
    def compute(self) -> Number:
        """
        Compute the metric.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the metric.
        """
        pass
