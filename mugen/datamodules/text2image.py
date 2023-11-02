from typing import Optional
import os.path as osp
from datasets import load_dataset, load_from_disk

import torch
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

from .base import BaseDataModule


class Text2ImageDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        cache_dir: str = ".cache",
        image_column: str = "image",
        caption_column: str = "caption",
        train_split: str = "train",
        val_split: str = "validation",
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        pipeline_name_or_path: str = None,
        vae_pretrained_name_or_path: str = None,
        tokenizer_pretrained_name_or_path: str = None,
        text_encoder_pretrained_name_or_path: str = None,
        load_cached: bool = True,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.data_path = data_path
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.image_column = image_column
        self.caption_column = caption_column
        self.train_split = train_split
        self.val_split = val_split
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.pipeline_name_or_path = pipeline_name_or_path
        self.vae_pretrained_name_or_path = vae_pretrained_name_or_path
        self.tokenizer_pretrained_name_or_path = tokenizer_pretrained_name_or_path
        self.text_encoder_pretrained_name_or_path = text_encoder_pretrained_name_or_path
        self.load_cached = load_cached
        self.batch_size = batch_size
        self.device = device

        self.save_path = osp.join(
            cache_dir, "_".join((data_path, data_name or "", "processed_text2img"))
        )
        self.train_save_path = osp.join(self.save_path, train_split)
        self.val_save_path = osp.join(self.save_path, val_split)

    def prepare_data(self):
        if (
            osp.exists(self.train_save_path)
            and osp.exists(self.val_save_path)
            and self.load_cached
        ):
            return

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
                transforms.Resize(
                    self.resolution, transforms.InterpolationMode.BILINEAR
                ),
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
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.vae_pretrained_name_or_path:
            vae = AutoencoderKL.from_pretrained(self.vae_pretrained_name_or_path)
        elif self.pipeline_name_or_path:
            vae = AutoencoderKL.from_pretrained(
                self.pipeline_name_or_path, subfolder="vae"
            )
        else:
            raise ValueError(
                "Either vae_pretrained_name_or_path or pipeline_name_or_path must be provided."
            )
        vae.to(device)

        def prepare_latent(examples):
            images = examples[self.image_column]
            images = [
                augs(img.convert("RGB")).unsqueeze(0).to(device) for img in images
            ]
            images = torch.cat(images, dim=0)
            z = vae.encode(images).latent_dist.sample().squeeze(0).cpu()

            examples["latent"] = z
            return examples

        print("Preparing latent vectors...")
        train_data = train_data.map(
            prepare_latent,
            batch_size=self.batch_size,
            writer_batch_size=1,
            batched=True,
        )
        val_data = val_data.map(
            prepare_latent,
            batch_size=self.batch_size,
            writer_batch_size=1,
            batched=True,
        )
        del vae

        if self.tokenizer_pretrained_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.tokenizer_pretrained_name_or_path
            )
        elif self.pipeline_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.pipeline_name_or_path, subfolder="tokenizer"
            )
        else:
            raise ValueError(
                "Either tokenizer_pretrained_name_or_path or pipeline_name_or_path must be provided."
            )

        if self.text_encoder_pretrained_name_or_path:
            text_encoder = CLIPTextModel.from_pretrained(
                self.text_encoder_pretrained_name_or_path
            )
        elif self.pipeline_name_or_path:
            text_encoder = CLIPTextModel.from_pretrained(
                self.pipeline_name_or_path, subfolder="text_encoder"
            )
        else:
            raise ValueError(
                "Either text_encoder_pretrained_name_or_path or pipeline_name_or_path must be provided."
            )
        text_encoder.to(device)

        def prepare_text_embedding(examples):
            captions = examples[self.caption_column]
            tokenized = tokenizer(
                captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            for k in tokenized:
                tokenized[k] = tokenized[k].to(device)

            embedding = text_encoder(**tokenized)[0].squeeze(0).cpu()
            examples["text_embedding"] = embedding
            return examples

        print("Preparing text embeddings...")
        train_data = train_data.map(
            prepare_text_embedding,
            batch_size=self.batch_size,
            writer_batch_size=1,
            batched=True,
        )
        val_data = val_data.map(
            prepare_text_embedding,
            batch_size=self.batch_size,
            writer_batch_size=1,
            batched=True,
        )
        del text_encoder, tokenizer

        train_data.set_format(type="torch", columns=["latent", "text_embedding"])
        val_data.set_format(type="torch", columns=["latent", "text_embedding"])

        train_data.save_to_disk(self.train_save_path)
        val_data.save_to_disk(self.val_save_path)

    def setup(self):
        if osp.exists(self.save_path):
            self.train_data = load_from_disk(self.train_save_path)
            self.val_data = load_from_disk(self.val_save_path)
        else:
            raise ValueError(
                "Data has not been prepared yet. Please run prepare_data() first."
            )

    def get_training_dataset(self):
        return self.train_data

    def get_validation_dataset(self):
        return self.val_data
