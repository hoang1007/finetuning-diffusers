from typing import Optional
from datasets import load_dataset

from torchvision import transforms


class ImageDataModule:
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
        self.train_data = load_dataset(
            data_path,
            name=data_name,
            cache_dir=cache_dir,
            split=train_split,
        )

        self.val_data = load_dataset(
            data_path,
            name=data_name,
            cache_dir=cache_dir,
            split=val_split,
        )

        augs = transforms.Compose(
            [
                transforms.Resize(resolution, transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip()
                if random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) if center_normalize else transforms.Lambda(lambda x: x),
            ]
        )

        def transform_images(examples):
            images = [augs(image.convert('RGB')) for image in examples[image_column]]
            ret = {'image': images}

            if caption_column is not None:
                ret['caption'] = examples[caption_column]

            return ret

        self.train_data.set_transform(transform_images)
        self.val_data.set_transform(transform_images)

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
