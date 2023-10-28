import pytest

from mugen.datamodules import LDMDataModule


@pytest.mark.parametrize(
    [
        "data_path",
        "data_name",
        "image_column",
        "train_split",
        "val_split",
        "resolution",
    ],
    [("cifar10", None, "img", "train[:2]", "test[2:4]", 32),
     ("lambdalabs/pokemon-blip-captions", None, "image", "train[:2]", "train[2:4]", 64)],
)
def test_get_data(
    data_path,
    data_name,
    image_column,
    train_split,
    val_split,
    resolution,
):
    dtm = LDMDataModule(
        data_path,
        data_name,
        image_column=image_column,
        train_split=train_split,
        val_split=val_split,
        resolution=resolution,
        vae_pretrained_name_or_path='stabilityai/sd-vae-ft-mse'
    )

    assert dtm.get_training_dataset() is not None
    assert dtm.get_validation_dataset() is not None

    assert len(dtm.get_training_dataset()) > 0
    assert len(dtm.get_validation_dataset()) > 0

    sample = dtm.get_training_dataset()[0]
    assert sample is not None
    assert sample['latent'] is not None
    assert sample['latent'].shape == (4, resolution // 8, resolution // 8)
