from typing import Optional
import os.path as osp
from datasets import load_dataset, load_from_disk, IterableDataset

import torch
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL


class Text2ImageDataModule:
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        cache_dir: str = '.cache',
        image_column: str = 'image',
        caption_column: str = 'caption',
        train_split: str = 'train',
        val_split: str = 'validation',
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        vae_pretrained_name_or_path: str = None,
        tokenizer_pretrained_name_or_path: str = None,
        text_encoder_pretrained_name_or_path: str = None,
        device: str = 'auto'
    ):
        save_path = osp.join(
            cache_dir, 
            '_'.join((data_path, data_name or '', 'processed_text2img'))
        )
        train_save_path = osp.join(save_path, train_split)
        val_save_path = osp.join(save_path, val_split)

        if osp.exists(save_path):
            self.train_data = load_from_disk(train_save_path)
            self.val_data = load_from_disk(val_save_path)
        else:
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
                del image
                return {'latent': z}
            
            print("Preparing latent vectors...")
            self.train_data = self.train_data.map(prepare_latent, writer_batch_size=1)
            self.val_data = self.val_data.map(prepare_latent, writer_batch_size=1)
            del vae

            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_pretrained_name_or_path)
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_pretrained_name_or_path)
            text_encoder.to(device)
            def prepare_text_embedding(example):
                caption = example[caption_column]
                tokenized = tokenizer(
                    [caption],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                )

                for k in tokenized:
                    tokenized[k] = tokenized[k].to(device)
                
                embedding = text_encoder(**tokenized)[0].squeeze(0).cpu()
                return {'text_embedding': embedding}

            print("Preparing text embeddings...")
            self.train_data = self.train_data.map(prepare_text_embedding, writer_batch_size=1)
            self.val_data = self.val_data.map(prepare_text_embedding, writer_batch_size=1)
            del text_encoder, tokenizer

            self.train_data.set_format(type='torch', columns=['latent', 'text_embedding'])
            self.val_data.set_format(type='torch', columns=['latent', 'text_embedding'])

            self.train_data.save_to_disk(train_save_path)
            self.val_data.save_to_disk(val_save_path)

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
