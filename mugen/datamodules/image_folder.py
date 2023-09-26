from typing import List, Dict, Optional, Callable, Any
from warnings import warn
from itertools import chain
from glob import glob
import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from .base import BaseDataModule


class CaptionImageDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], transform: Optional[Callable] = None):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        img = Image.open(item['img_path'])
        if self.transform is not None:
            img = self.transform(img)
        return dict(image=img, prompt=item['prompt'])


class CaptionImageFolderDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        train_ratio: float = 0.75,
    ):
        self.data_dir = data_dir
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

        dataset = CaptionImageDataset(self.__prepare(), augs)
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            (train_ratio, 1 - train_ratio),
            torch.Generator().manual_seed(0)
        )

    def get_training_dataset(self):
        return self.train_dataset
    
    def get_validation_dataset(self):
        return self.val_dataset
    
    def __prepare(self) -> List[Dict[str, Any]]:
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        data = dict()

        for img_path in chain(*[glob(f"*{img_ext}", root_dir=self.data_dir) for img_ext in img_exts]):
            img_id = osp.splitext(img_path)[0]
            img_path = osp.join(self.data_dir, img_path)
            data[img_id] = dict(img_path=img_path)
        
        for prompt_path in glob("*.txt", root_dir=self.data_dir):
            img_id = osp.splitext(prompt_path)[0]
            prompt_path = osp.join(self.data_dir, prompt_path)
            if img_id in data:
                with open(prompt_path, 'r', encoding='utf8') as f:
                    data[img_id]['prompt'] = f.read()
            else:
                warn(f"Prompt {img_id} does not associated with any images!")

        remove_ids = []
        for img_id in data:
            if 'prompt' not in data[img_id]:
                warn(f"Image {img_id} does not have prompt. Removing...")
                remove_ids.append(img_id)
            if 'img_path' not in data[img_id]:
                warn(f"Prompt {img_id} does not associated with any images. Removing...")
                remove_ids.append(img_id)
        for img_id in remove_ids:
            data.pop(img_id)
        
        return list(data.values())
