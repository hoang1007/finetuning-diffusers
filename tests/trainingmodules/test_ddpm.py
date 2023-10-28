import pytest

import torch
from mugen.trainingmodules import DDPMTrainingModule

@pytest.mark.parametrize(
    "batch",
    [
        {
            "image": torch.rand(1, 3, 32, 32),
        },
        {
            "image": torch.rand(2, 3, 32, 32),
        },
    ]
)
def test_forward(batch):
    module = DDPMTrainingModule(
        unet_config=dict(
            sample_size=32,
            block_out_channels=[32, 32, 32, 32]
        ),
        scheduler_config=dict()
    )

    output = module.training_step(batch, 0, 0)
    assert isinstance(output, torch.Tensor)
