from typing import Optional
import os.path as osp
from datasets import load_dataset, load_from_disk

import torch
from torchvision import transforms

from diffusers import AutoencoderKL

from .base import BaseDataModule


class LDMDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        cache_dir: str = '.cache',
        image_column: str = 'image',
        train_split: str = 'train',
        val_split: str = 'validation',
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        vae_pretrained_name_or_path: str = None,
        device: str = 'auto'
    ):
        self.data_path = data_path
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.image_column = image_column
        self.train_split = train_split
        self.val_split = val_split
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.vae_pretrained_name_or_path = vae_pretrained_name_or_path
        self.device = device
    
    def _prepare_data(self):
        train_data = load_dataset(
            self.data_path,
            name=self.data_name,
            cache_dir=self.cache_dir,
            split=self.train_split,
        )

        val_data = load_dataset(
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
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        vae = AutoencoderKL.from_pretrained(self.vae_pretrained_name_or_path)
        vae.to(device)

        def prepare_latent(example):
            image = example[self.image_column]
            image = augs(image.convert('RGB')).unsqueeze(0).to(device)
            z = vae.encode(image).latent_dist.sample().squeeze(0).cpu()
            return {'latent': z}
        
        print("Preparing latent vectors...")
        train_data = train_data.map(prepare_latent, writer_batch_size=1)
        val_data = val_data.map(prepare_latent, writer_batch_size=1)

        train_data.set_format(type='torch', columns=['latent'])
        val_data.set_format(type='torch', columns=['latent'])
        
        return train_data, val_data

    def prepare_data(self):
        self._prepare_data()

    def setup(self):
        self.train_data, self.val_data = self._prepare_data()

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
