import pytest

from mugen.datamodules import Text2ImageDataModule


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
    [("lambdalabs/pokemon-blip-captions", None, "image", "text", "train[:5%]", "train[95%:]", 64)],
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
    dtm = Text2ImageDataModule(
        data_path,
        data_name,
        image_column=image_column,
        caption_column=caption_column,
        train_split=train_split,
        val_split=val_split,
        resolution=resolution,
        pipeline_name_or_path='CompVis/stable-diffusion-v1-4'
    )

    assert dtm.get_training_dataset() is not None
    assert dtm.get_validation_dataset() is not None

    assert len(dtm.get_training_dataset()) > 0
    assert len(dtm.get_validation_dataset()) > 0

    sample = dtm.get_training_dataset()[0]
    assert sample is not None
    assert sample['latent'] is not None
    assert sample['latent'].shape == (4, resolution // 8, resolution // 8)

    if caption_column is not None:
        assert sample['text_embedding'] is not None
        assert len(sample['text_embedding'].shape) == 2
        assert sample['text_embedding'].shape[-1] == 768
