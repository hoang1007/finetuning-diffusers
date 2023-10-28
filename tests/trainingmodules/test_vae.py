import pytest

import torch
from mugen.trainingmodules import VAETrainingModule
from mugen.losses.lpips import is_lpips_available


@pytest.mark.skipif(not is_lpips_available(), reason="LPIPS is not installed")
@pytest.mark.parametrize(
    "batch",
    [
        {
            "image": torch.randn(1, 3, 64, 64),
        }
    ],
)
def test_forward_lpips(batch):
    module = VAETrainingModule(
        pretrained_name_or_path="stabilityai/sd-vae-ft-mse",
        lpips_config=dict(disc_start=0),
    )

    output = module.training_step(batch, 0, 0)
    assert isinstance(output, torch.Tensor)

    output = module.training_step(batch, 0, 1)
    assert isinstance(output, torch.Tensor)


def test_lpips_not_avaiable_error():
    with pytest.raises(AssertionError):
        module = VAETrainingModule(
            pretrained_name_or_path="stabilityai/sd-vae-ft-mse",
            lpips_config=dict(disc_start=0),
            use_ema=False,
        )


@pytest.mark.parametrize(
    "batch",
    [
        {
            "image": torch.randn(1, 3, 64, 64),
        }
    ],
)
def test_forward_mse(batch):
    module = VAETrainingModule(
        pretrained_name_or_path="stabilityai/sd-vae-ft-mse",
    )

    output = module.training_step(batch, 0, 0)
    assert isinstance(output, torch.Tensor)
