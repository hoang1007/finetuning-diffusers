import pytest

import torch
from mugen.trainingmodules import CLIPTrainingModule


@pytest.mark.parametrize(
    "batch",
    [
        {
            "image": torch.rand(1, 3, 224, 224),
            "caption": "a photo of a cat",
        },
        {
            "image": torch.rand(2, 3, 224, 224),
            "caption": ["a photo of a cat", "a photo of a dog"],
        },
    ]
)
def test_forward(batch):
    module = CLIPTrainingModule(pretrained_name_or_path="openai/clip-vit-base-patch32")

    output = module.training_step(batch, 0, 0)
    assert isinstance(output, torch.Tensor)
