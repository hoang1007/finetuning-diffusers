import pytest

import torch
from mugen.trainingmodules import LDMTrainingModule


@pytest.mark.parametrize(
    "batch",
    [
        {
            "latent": torch.rand(1, 4, 16, 16),
        },
    ]
)
def test_forward(batch):
    module = LDMTrainingModule(
        unet_config=dict(
            sample_size=16,
            block_out_channels=[32, 32, 32, 32]
        ),
        vae_pretrained_name_or_path='stabilityai/sd-vae-ft-mse',
        scheduler_config=dict()
    )

    with pytest.raises(Exception):
        output = module.training_step(batch, 0, 0)
        assert isinstance(output, torch.Tensor)
