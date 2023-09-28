from typing import Dict, Union, Any
from omegaconf import DictConfig, OmegaConf
import importlib


def get_config(config_path: str) -> DictConfig:
    """Get the config from a yaml file.

    Args:
        config_path (str): path to the config file

    Returns:
        DictConfig: the config
    """
    return OmegaConf.load(config_path)


def init_from_config(config: Dict[str, Any], reload: bool = False):
    """Initialize object from config."""
    cfg = config.copy()
    target_key = '_target_'
    assert target_key in cfg, f'Key {target_key} is required for object initialization!'

    module, cls = cfg.pop(target_key).rsplit('.', 1)
    module = importlib.import_module(module)
    if reload:
        module = importlib.reload(module)
    return getattr(module, cls)(**cfg)
