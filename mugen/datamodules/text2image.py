from typing import Optional
import os.path as osp
from datasets import load_dataset, load_from_disk

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
        pipeline_name_or_path: str = None,
        vae_pretrained_name_or_path: str = None,
        tokenizer_pretrained_name_or_path: str = None,
        text_encoder_pretrained_name_or_path: str = None,
        load_cached: bool = True,
        batch_size: int = 32,
        device: str = 'auto'
    ):
        save_path = osp.join(
            cache_dir, 
            '_'.join((data_path, data_name or '', 'processed_text2img'))
        )
        train_save_path = osp.join(save_path, train_split)
        val_save_path = osp.join(save_path, val_split)

        if osp.exists(save_path) and load_cached:
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

            if vae_pretrained_name_or_path:
                vae = AutoencoderKL.from_pretrained(vae_pretrained_name_or_path)
            elif pipeline_name_or_path:
                vae = AutoencoderKL.from_pretrained(pipeline_name_or_path, subfolder='vae')
            else:
                raise ValueError("Either vae_pretrained_name_or_path or pipeline_name_or_path must be provided.")
            vae.to(device)

            def prepare_latent(examples):
                images = examples[image_column]
                images = [augs(img.convert('RGB')).unsqueeze(0).to(device) for img in images]
                images = torch.cat(images, dim=0)
                z = vae.encode(images).latent_dist.sample().squeeze(0).cpu()

                examples['latent'] = z
                return examples
            
            print("Preparing latent vectors...")
            self.train_data = self.train_data.map(prepare_latent, batch_size=batch_size, writer_batch_size=1, batched=True)
            self.val_data = self.val_data.map(prepare_latent, batch_size=batch_size, writer_batch_size=1, batched=True)
            del vae

            if tokenizer_pretrained_name_or_path:
                tokenizer = CLIPTokenizer.from_pretrained(tokenizer_pretrained_name_or_path)
            elif pipeline_name_or_path:
                tokenizer = CLIPTokenizer.from_pretrained(pipeline_name_or_path, subfolder='tokenizer')
            else:
                raise ValueError("Either tokenizer_pretrained_name_or_path or pipeline_name_or_path must be provided.")

            if text_encoder_pretrained_name_or_path:
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_pretrained_name_or_path)
            elif pipeline_name_or_path:
                text_encoder = CLIPTextModel.from_pretrained(pipeline_name_or_path, subfolder='text_encoder')
            else:
                raise ValueError("Either text_encoder_pretrained_name_or_path or pipeline_name_or_path must be provided.")
            text_encoder.to(device)
            def prepare_text_embedding(examples):
                captions = examples[caption_column]
                tokenized = tokenizer(
                    captions,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                )

                for k in tokenized:
                    tokenized[k] = tokenized[k].to(device)
                
                embedding = text_encoder(**tokenized)[0].squeeze(0).cpu()
                examples['text_embedding'] = embedding
                return examples

            print("Preparing text embeddings...")
            self.train_data = self.train_data.map(prepare_text_embedding, batch_size=batch_size, writer_batch_size=1, batched=True)
            self.val_data = self.val_data.map(prepare_text_embedding, batch_size=batch_size, writer_batch_size=1, batched=True)
            del text_encoder, tokenizer

            self.train_data.set_format(type='torch', columns=['latent', 'text_embedding'])
            self.val_data.set_format(type='torch', columns=['latent', 'text_embedding'])

            self.train_data.save_to_disk(train_save_path)
            self.val_data.save_to_disk(val_save_path)

    def get_training_dataset(self):
        return self.train_data
    
    def get_validation_dataset(self):
        return self.val_data
