import pytest

from mugen.datamodules import ImageDataModule


@pytest.mark.parametrize(
    [
        "data_path",
        "data_name",
        "image_column",
        "caption_column",
        "train_split",
        "val_split",
        "resolution",
    ],
    [("cifar10", None, "img", None, "train[:2]", "test[2:4]", 32),
     ("lambdalabs/pokemon-blip-captions", None, "image", "text", "train[:2]", "train[2:4]", 64)],
)
def test_get_data(
    data_path,
    data_name,
    image_column,
    caption_column,
    train_split,
    val_split,
    resolution,
):
    dtm = ImageDataModule(
        data_path,
        data_name,
        image_column,
        caption_column,
        train_split=train_split,
        val_split=val_split,
        resolution=resolution,
    )

    assert dtm.get_training_dataset() is not None
    assert dtm.get_validation_dataset() is not None

    assert len(dtm.get_training_dataset()) > 0
    assert len(dtm.get_validation_dataset()) > 0

    sample = dtm.get_training_dataset()[0]
    assert sample is not None
    assert sample['image'] is not None
    assert sample['image'].shape == (3, resolution, resolution)

    if caption_column is not None:
        assert sample['caption'] is not None
        assert isinstance(sample['caption'], str)
