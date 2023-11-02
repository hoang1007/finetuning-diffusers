import os
from omegaconf import OmegaConf

from mugen.datamodules import BaseDataModule
from mugen.trainer import Trainer
from mugen.training_args import TrainingArguments
from mugen.trainingmodules.base import TrainingModule
from mugen.utils.config_utils import init_from_config


def load_config():
    config = OmegaConf.from_cli()

    if 'config' in config:
        base_config = OmegaConf.load(config.config)
        config = OmegaConf.merge(base_config, config)

    config_name = os.path.splitext(os.path.basename(config.config))[0]
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("project_name") is None:
        config['project_name'] = config_name
    config['training_args']['output_dir'] = os.path.join(config['training_args']['output_dir'], config_name)

    return config


def main(config: dict):
    training_args = TrainingArguments(**config['training_args'])
    data_module: BaseDataModule = init_from_config(config['datamodule'])
    training_module: TrainingModule = init_from_config(config['training_module'])

    trainer = Trainer(
        project_name="finetuning-diffusers",
        training_module=training_module,
        training_args=training_args,
        data_module=data_module
    )
    trainer.get_tracker().store_init_configuration(config)
    trainer.start()


if __name__ == "__main__":
    config = load_config()
    main(config)
