from typing import Optional
from datasets import load_dataset

from torchvision import transforms

from lightning_accelerate import DataModule


class ImageDataModule(DataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        image_column: str = "image",
        cache_dir: str = ".cache",
        train_split: str = "train",
        val_split: str = "validation",
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        center_normalize: bool = True,
    ):
        super().__init__()

    def setup(self):
        self.train_data = load_dataset(
            self.config.data_path,
            name=self.config.data_name,
            cache_dir=self.config.cache_dir,
            split=self.config.train_split,
        )

        self.val_data = load_dataset(
            self.config.data_path,
            name=self.config.data_name,
            cache_dir=self.config.cache_dir,
            split=self.config.val_split,
        )

        augs = transforms.Compose(
            [
                transforms.Resize(
                    self.config.resolution, transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.config.resolution)
                if self.config.center_crop
                else transforms.RandomCrop(self.config.resolution),
                transforms.RandomHorizontalFlip()
                if self.config.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                if self.config.center_normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        def transform_images(examples):
            images = [
                augs(image) for image in examples[self.config.image_column]
            ]
            ret = {"image": images}
            return ret

        self.train_data.set_transform(transform_images)
        self.val_data.set_transform(transform_images)

    def get_training_dataset(self):
        return self.train_data

    def get_validation_dataset(self):
        return self.val_data
