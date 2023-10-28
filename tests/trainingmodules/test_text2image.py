import pytest

import torch
from mugen.trainingmodules import Text2ImageTrainingModule

@pytest.mark.skip(reason="Test runs too long")
@pytest.mark.parametrize(
    "batch",
    [
        {
            "latent": torch.rand(1, 4, 16, 16),
            "text_embedding": torch.rand(1, 768)
        },
    ]
)
def test_forward(batch):
    module = Text2ImageTrainingModule('CompVis/stable-diffusion-v1-4')

    with pytest.raises(Exception):
        output = module.training_step(batch, 0, 0)
        assert isinstance(output, torch.Tensor)
