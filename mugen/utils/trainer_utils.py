import os
import shutil
import re

import random
import numpy as np
import torch

from accelerate import Accelerator


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy` and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_using_gpu(accelerator: Accelerator):
    """
    Helper function to check if GPU is available.

    Returns:
        `bool`: `True` if GPU is available, `False` otherwise.
    """
    return accelerator.device.type == "cuda"


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder: str):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def prune_checkpoints(folder: str, limit: int):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) <= limit:
        return
    checkpoints.sort(key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    for checkpoint in checkpoints[:-limit]:
        checkpoint_path = os.path.join(folder, checkpoint)
        shutil.rmtree(checkpoint_path)
