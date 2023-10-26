from torch import nn
from mugen.trainingmodules.base import TrainingModule


def unwrap_model(model):
    if isinstance(model, DDPWrapper):
        return model.training_module
    elif not isinstance(model, TrainingModule):
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            raise ValueError(f"Unrecognized model type!. Got {model.__class__.__name__}")
    return model


class DDPWrapper(nn.Module):
    def __init__(self, training_module: TrainingModule):
        super().__init__()
        self.training_module = training_module

    def forward(self, *args, **kwargs):
        if self.training:
            self.training_module.training_step(*args, **kwargs)
        else:
            self.training_module.validation_step(*args, **kwargs)
