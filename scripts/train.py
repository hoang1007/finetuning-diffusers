from lightning_accelerate import Trainer, TrainingArguments, DataModule, TrainingModule
from lightning_accelerate.utils.config_utils import parse_config_from_cli, init_from_config


def main(config: dict):
    training_args = TrainingArguments(**config['training_args'])
    data_module: DataModule = init_from_config(config['datamodule'])
    training_module: TrainingModule = init_from_config(config['training_module'])

    Trainer(
        project_name="finetuning-diffusers",
        training_module=training_module,
        training_args=training_args,
        data_module=data_module
    ).fit()


if __name__ == "__main__":
    config = parse_config_from_cli()
    main(config)
