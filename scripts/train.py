from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig

from mugen.datasets import BaseDataModule
from mugen.trainers.base_trainer import Trainer, TrainingArguments, TrainingWrapper
from mugen.utils.config import init_from_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    return parser.parse_args()


def load_config(args):
    config = OmegaConf.load(args.config)
    return config


def main(config: DictConfig):
    training_args = TrainingArguments(**config.training_args)
    datamodule: BaseDataModule = init_from_config(config.datamodule)
    training_wrapper: TrainingWrapper = init_from_config(config.training_wrapper)

    trainer = Trainer(
        training_wrapper,
        training_args,
        datamodule.get_training_dataset(),
        datamodule.get_validation_dataset(),
    )

    trainer.start()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    main(config)
