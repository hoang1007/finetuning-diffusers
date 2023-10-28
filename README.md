[![codecov](https://codecov.io/gh/hoang1007/finetuning-diffusers/graph/badge.svg?token=S7VKN050RD)](https://codecov.io/gh/hoang1007/finetuning-diffusers)
[![CodeQL](https://github.com/hoang1007/finetuning-diffusers/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/hoang1007/finetuning-diffusers/actions/workflows/github-code-scanning/codeql)

Finetuning-diffusers aim to provide a simple way to finetune pretrained diffusion models on your own data. This repository is built on top of [diffusers](https://github.com/huggingface/diffusers.git), is designed to be easy to use and maximize customizability.

Our project is highly insprired by [HuggingFace Transformers](https://github.com/huggingface/transformers.git) and [PyTorch Lightning](https://github.com/Lightning-AI/lightning.git).

# Table of contents
<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [Features](#features)
- [Setup](#setup)
- [Understanding the code](#understanding-the-code)
  - [What's `TrainingModule`?](#whats-trainingmodule)
  - [`DataModule`:](#datamodule)
  - [`Trainer` and `TrainingArguments`](#trainer-and-trainingarguments)
- [Training](#training)
    - [Training with DeepSpeed](#training-with-deepspeed)

<!-- TOC end -->


# Features
- [x] Finetune on your own data
- [x] Memory efficiency training with DeepSpeed, bitsandbytes integrated
- [x] Finetune diffusion models with LoRA

# Setup
Just only clone and install this repository:
```bash
git clone https://github.com/hoang1007/finetuning-diffusers.git
pip install -v -e .
```

# Understanding the code
To training or finetuning models, you need to prepare `TrainingModule` and `DataModule`.

## What's `TrainingModule`?
`TrainingModule` is a base class that cover the logic of training and evaluation. It is designed to be easy to use and maximize customizability. Let's take a look at the following example:
```python
class DDPMTrainingModule(TrainingModule):
    def __init__(self, pretrained_name_or_path: Optional[str] = None):
        super().__init__()

        self.unet = UNet2DModel.from_pretrained(pretrained_name_or_path, subfolder="unet")
    
        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_name_or_path, subfolder="scheduler")

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        """This function calculate the loss of the model to optimize"""
        x = batch
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (x.size(0),),
            device=x.device,
        ).long()

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        unet_output = self.unet(noisy_x, timesteps).sample

        loss = F.mse_loss(unet_output, noise)
        self.log({"train/loss": loss.item()})

        return loss
    
    def on_validation_epoch_start(self):
        self.random_batch_idx = torch.randint(
            0, len(self.trainer.val_dataloader), (1,)
        ).item()

    def validation_step(self, batch, batch_idx: int):
        # Only log one batch per epoch
        if batch_idx != self.random_batch_idx:
            return
        
        x = batch

        pipeline = DDIMPipeline(self.unet, self.noise_scheduler)

        images = pipeline(
            batch_size=x.size(0),
            output_type='np'
        ).images

        org_imgs = (x.detach() / 2 + 0.5).cpu().permute(0, 2, 3, 1).numpy()

        self.log_images(
            {
                "original": org_imgs,
                "generated": images,
            }
        )

    def get_optim_params(self) -> List[Iterable[torch.nn.Parameter]]:
        """Return a list of param group. Each group will be handled by an optimizer"""
        return [self.unet.parameters()]

    def save_pretrained(self, output_dir):
        """Save model's weights to load later"""
        DDIMPipeline(self.unet, self.noise_scheduler).save_pretrained(output_dir)

```
Our `TrainingModule` is designed to familiar with `Pytorch LightningModule`. For more details, please take a look at [TrainingModule API](mugen/trainingmodules/base.py)

Currently, we already implemented some `TrainingModule` to training diffusion tasks:
- [DDPMTrainingModule](mugen/trainingmodules/ddpm.py): For unconditional image generation.
- [CLIPTrainingModule](mugen/trainingmodules/clip.py): Training CLIP text model for text2image generation.
- [VAETrainingModule](mugen/trainingmodules/vae.py): Training VAE model.
- [Text2ImageTrainingModule](mugen/trainingmodules/text2image.py): For training Stable Diffusion.

## `DataModule`:
`DataModule` is a base class that prepare data for training and evaluation step.

## `Trainer` and `TrainingArguments`
`Trainer` is a class that contains all the logic for training and evaluation. `TrainingArguments` contains all the hyperparameters that you can pass to `Trainer` to control the training. For more details, please take a look at [TrainingArguments](mugen/training_args.py).

# Training
To train your model, create a config file and run the following command:
```bash
python scripts/train.py <path_to_your_config_file>
```

An example of config file for training DDPM model on CIFAR10 dataset:
```yaml
training_args: # Config for TrainingArguments
  output_dir: outputs
  num_epochs: 100

  learning_rate: 1e-4
  lr_scheduler_type: constant_with_warmup
  warmup_steps: 1000

  mixed_precision: fp16

  train_batch_size: 256
  eval_batch_size: 64
  data_loader_num_workers: 8

  logger: wandb
  save_steps: 1000
  save_total_limit: 3
  resume_from_checkpoint: latest

  tracker_init_kwargs:
    group: 'ddpm'

training_module: # Used to instantiate TrainingModule
  _target_: mugen.trainingmodules.ddpm.DDPMTrainingModule
  # pretrained_name_or_path: google/ddpm-cifar10-32
  unet_config:
    sample_size: 32
    block_out_channels: [128, 256, 256, 256]
    layers_per_block: 2
  scheduler_config:
    num_train_timesteps: 1000
  use_ema: true
  enable_xformers_memory_efficient_attention: true
  enable_gradient_checkpointing: true

datamodule: # Used to instantiate DataModule
  _target_: mugen.datamodules.ImageDataModule
  dataset_name: cifar10
  train_split: train
  val_split: test
  image_column: img
  resolution: 32
```

Or you can manually prepare `TrainingModule` and `DataModule`. Then, pass them to `Trainer` to train your model:
```python
from mugen import Trainer, TrainingArguments

args = TrainingArguments(...)
Trainer(
    "finetuning-diffusers",
    training_module,
    args,
    data_module.get_training_dataset(),
    data_mdoule.get_validation_dataset())
.start()
```

### Training with DeepSpeed
To training with DeepSpeed, please refer to [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
