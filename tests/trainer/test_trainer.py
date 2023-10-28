import pytest
import tempfile

from mugen import Trainer, TrainingArguments
from mugen.datamodules import ImageDataModule
from mugen.trainingmodules import DDPMTrainingModule


def test_trainer_train():
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir, train_batch_size=2, eval_batch_size=1, num_epochs=1
        )

        tm = DDPMTrainingModule(
            unet_config=dict(sample_size=32, block_out_channels=[32, 32, 32, 32]),
            scheduler_config=dict(),
        )

        dtm = ImageDataModule(
            "cifar10",
            train_split="train[:2]",
            val_split="test[2:4]",
            image_column='img',
            resolution=32,
        )

        trainer = Trainer(
            "temp", tm, args, dtm.get_training_dataset(), dtm.get_validation_dataset()
        )
        trainer.start()


def test_trainer_load_ckpt():
    print("BRUH")
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            train_batch_size=2,
            eval_batch_size=1,
            num_epochs=1,
            save_steps=1,
            save_total_limit=1,
            resume_from_checkpoint='latest'
        )
        tm = DDPMTrainingModule(
            unet_config=dict(sample_size=32, block_out_channels=[32, 32, 32, 32]),
            scheduler_config=dict(),
        )
        dtm = ImageDataModule(
            "cifar10",
            train_split="train[:4]",
            val_split="test[4:8]",
            image_column='img',
            resolution=32,
        )
        trainer = Trainer(
            "temp", tm, args, dtm.get_training_dataset(), dtm.get_validation_dataset()
        )
        trainer.start()
        trainer.accelerator.load_state()
