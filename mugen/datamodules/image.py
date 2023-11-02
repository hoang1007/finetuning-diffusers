from typing import Optional
from datasets import load_dataset

from torchvision import transforms

from .base import BaseDataModule


class ImageDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        image_column: str = 'image',
        caption_column: Optional[str] = None,
        cache_dir: str = '.cache',
        train_split: str = 'train',
        val_split: str = 'validation',
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        center_normalize: bool = True,
    ):
        self.data_path = data_path
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.train_split = train_split
        self.val_split = val_split
        self.image_column = image_column
        self.caption_column = caption_column
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.center_normalize = center_normalize

    def setup(self):
        self.train_data = load_dataset(
            self.data_path,
            name=self.data_name,
            cache_dir=self.cache_dir,
            split=self.train_split,
        )

        self.val_data = load_dataset(
            self.data_path,
            name=self.data_name,
            cache_dir=self.cache_dir,
            split=self.val_split,
        )

        augs = transforms.Compose(
            [
                transforms.Resize(self.resolution, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution)
                if self.center_crop
                else transforms.RandomCrop(self.resolution),
                transforms.RandomHorizontalFlip()
                if self.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) if self.center_normalize else transforms.Lambda(lambda x: x),
            ]
        )

        def transform_images(examples):
            images = [augs(image.convert('RGB')) for image in examples[self.image_column]]
            ret = {'image': images}

            if self.caption_column is not None:
                ret['caption'] = examples[self.caption_column]

            return ret

        self.train_data.set_transform(transform_images)
        self.val_data.set_transform(transform_images)

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
