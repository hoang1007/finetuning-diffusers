from typing import Optional
import os.path as osp
from datasets import load_dataset, load_from_disk

import torch
from torchvision import transforms

from diffusers import AutoencoderKL


class LDMDataModule:
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
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        vae = AutoencoderKL.from_pretrained(vae_pretrained_name_or_path)
        vae.to(device)

        def prepare_latent(example):
            image = example[image_column]
            image = augs(image.convert('RGB')).unsqueeze(0).to(device)
            z = vae.encode(image).latent_dist.sample().squeeze(0).cpu()
            return {'latent': z}
        
        print("Preparing latent vectors...")
        self.train_data = self.train_data.map(prepare_latent, writer_batch_size=1)
        self.val_data = self.val_data.map(prepare_latent, writer_batch_size=1)

        self.train_data.set_format(type='torch', columns=['latent'])
        self.val_data.set_format(type='torch', columns=['latent'])

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
