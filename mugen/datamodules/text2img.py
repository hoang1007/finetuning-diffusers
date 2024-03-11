from tkinter.tix import IMAGE
from typing import Optional, Tuple, Union
import os

from datasets import load_dataset, Dataset
from datasets import Image as ImageFeature

import torch
from torchvision import transforms
import random

from diffusers import AutoencoderKL
from lightning_accelerate import DataModule

from mugen.loaders import load_pipeline


LATENT_COLUMN = "latent"
IMAGE_COLUMN = "image"
CAPTION_COLUMN = "text"


class BaseText2ImageDataModule(DataModule):
    def __init__(
        self,
        image_column: str = "image",
        caption_column: str = "caption",
        shuffle_tags: bool = False,
        resolution: int = 256,
        center_crop: bool = True,
        random_flip: bool = True,
        cache_latents: bool = True,
        pipeline_name_or_path: Optional[str] = None,
        is_from_original_sd: bool = False,
        vae_pretrained_name_or_path: Optional[str] = None,
        batch_size: int = 4,
        device: str = "auto",
    ):
        super().__init__()

    def load_dataset(self) -> Tuple[Dataset, Dataset]:
        pass

    def _prepare_data(self):
        train_data, val_data = self.load_dataset()
        train_data = train_data.rename_columns(
            {
                self.config.image_column: IMAGE_COLUMN,
                self.config.caption_column: CAPTION_COLUMN,
            }
        )
        val_data = val_data.rename_columns(
            {
                self.config.image_column: IMAGE_COLUMN,
                self.config.caption_column: CAPTION_COLUMN,
            }
        )

        augs = transforms.Compose(
            [
                transforms.Resize(
                    self.config.resolution, transforms.InterpolationMode.BILINEAR
                ),
                # transforms.CenterCrop(self.config.resolution)
                # if self.config.center_crop
                # else transforms.RandomCrop(self.config.resolution),
                # transforms.RandomHorizontalFlip()
                # if self.config.random_flip
                # else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def aug_captions(examples):
            tags = examples[CAPTION_COLUMN]
            tags = [t.strip() for t in tags.split(",")]
            random.shuffle(tags)
            examples[CAPTION_COLUMN] = ", ".join(tags)
            return examples

        # Cache latents if VAE is provided
        if self.config.cache_latents:
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.config.vae_pretrained_name_or_path is not None:
                vae = AutoencoderKL.from_pretrained(
                    self.config.vae_pretrained_name_or_path
                )
            elif self.config.pipeline_name_or_path is not None:
                vae = load_pipeline(
                    self.config.pipeline_name_or_path, self.config.is_from_original_sd
                ).vae
            else:
                raise ValueError(
                    "Either `vae_pretrained_name_or_path` or `pipeline_name_or_path` must be provided to use the caching latents feature."
                )

            vae.to(device)

            def prepare_latent(examples):
                images = examples[IMAGE_COLUMN]
                images = [augs(img.convert("RGB")).to(device) for img in images]
                images = torch.stack(images, dim=0)
                z = vae.encode(images).latent_dist.sample().detach().cpu()

                examples[LATENT_COLUMN] = z
                return examples

            print("Preparing latent vectors...")
            train_data = train_data.map(
                prepare_latent,
                batch_size=self.config.batch_size,
                writer_batch_size=1,
                batched=True,
                remove_columns=[
                    col
                    for col in train_data.column_names
                    if col not in (LATENT_COLUMN, CAPTION_COLUMN)
                ],
            )
            val_data = val_data.map(
                prepare_latent,
                batch_size=self.config.batch_size,
                writer_batch_size=1,
                batched=True,
                remove_columns=[
                    col
                    for col in val_data.column_names
                    if col not in (LATENT_COLUMN, CAPTION_COLUMN)
                ],
            )

            if self.config.shuffle_tags:
                train_data.set_transform(aug_captions)
        else:

            def transforms_fn(examples, shuffle_tags=False):
                ret = dict()
                ret[IMAGE_COLUMN] = [
                    augs(image.convert("RGB")) for image in examples[IMAGE_COLUMN]
                ]

                ret[CAPTION_COLUMN] = examples[CAPTION_COLUMN]
                if shuffle_tags:
                    ret = aug_captions(ret)

                return ret

            train_data.set_transform(lambda x: transforms_fn(x, shuffle_tags=True))
            val_data.set_transform(lambda x: transforms_fn(x, shuffle_tags=False))

        train_data.set_format("torch")
        val_data.set_format("torch")
        return train_data, val_data

    def prepare_data(self):
        return
        self._prepare_data()

    def setup(self):
        self.train_data, self.val_data = self._prepare_data()

    def get_training_dataset(self):
        return self.train_data

    def get_validation_dataset(self):
        return self.val_data


class Text2ImageDataModule(BaseText2ImageDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        data_name: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "validation",
        cache_dir: str = ".cache",
        **kwargs,
    ):
        super().__init__(**kwargs)

    def load_dataset(self):
        train_data = load_dataset(
            self.config.data_path,
            name=self.config.data_name,
            cache_dir=self.config.cache_dir,
            split=self.config.train_split,
        )

        val_data = load_dataset(
            self.config.data_path,
            name=self.config.data_name,
            cache_dir=self.config.cache_dir,
            split=self.config.val_split,
        )

        return train_data, val_data


class Text2ImageFolderDataModule(BaseText2ImageDataModule):
    def __init__(
        self,
        data_path: str,
        cache_dir: str = ".cache",
        test_size: Union[float, int] = 0.25,
        **kwargs,
    ):
        super().__init__(image_column="image", caption_column="caption", **kwargs)

    def load_dataset(self):
        data = dict(image=[], caption=[])
        for f in os.scandir(self.config.data_path):
            if os.path.splitext(f.name)[-1] in (".jpg", ".png", ".jpeg"):
                data["image"].append(f.path)
        for impath in data["image"]:
            tags_path = os.path.splitext(impath)[0] + ".txt"
            tags = open(tags_path, encoding="utf-8").read().strip()
            data["caption"].append(tags)

        data = Dataset.from_dict(data).cast_column("image", ImageFeature())
        train_data, val_data = data.train_test_split(
            test_size=self.config.test_size, shuffle=False
        ).values()

        return train_data, val_data
