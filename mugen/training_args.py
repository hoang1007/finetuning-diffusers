from typing import Literal, Optional
from dataclasses import asdict, dataclass, field, fields
import os

import math

from accelerate.utils import ProjectConfiguration, DeepSpeedPlugin


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    logging_dir: Optional[str] = field(
        default="logs", metadata={"help": "Logging directory."}
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )

    use_cpu: bool = field(default=False, metadata={"help": "Use CPU."})

    train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for training."}
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )

    eval_steps: float = field(
        default=500,
        metadata={
            "help": "Run an evaluation every X steps. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps."
        },
    )

    data_loader_num_workers: int = field(
        default=4, metadata={"help": "Number of workers for data loader."}
    )
    num_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm"}
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for optimizers."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for Adam optimizer"}
    )
    adam_weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for Adam optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
    )
    use_8bit_adam: bool = field(
        default=False, metadata={"help": "Use 8bit Adam optimizer from bitsandbytes."}
    )

    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = field(
        default="constant_with_warmup",
        metadata={"help": "The type of learning rate scheduler to use."},
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio."}
    )

    use_ema: bool = field(
        default=True, metadata={"help": "Use Exponential Moving Average when training."}
    )
    ema_inv_gamma: float = field(default=1.0, metadata={"help": "EMA inverse gamma."})
    ema_power: float = field(default=3 / 4, metadata={"help": "EMA power."})
    ema_max_decay: float = field(default=0.9999, metadata={"help": "EMA max decay."})

    logger: Literal["tensorboard", "wandb"] = field(
        default="wandb", metadata={"help": "Logger to use."}
    )

    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )

    mixed_precision: Literal["no", "fp16", "bf16"] = field(
        default="no", metadata={"help": "Mixed precision training."}
    )

    use_lora: bool = field(
        default=False, metadata={"help": "Finetuning with LoRA."}
    )
    lora_rank: int = field(
        default=8, metadata={"help": "LoRA rank, only used if use_lora is True."}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha, only used if use_lora is True."}
    )
    lora_rank_dropout: float = field(
        default=0.0, metadata={"help": "LoRA rank dropout, only used if use_lora is True."}
    )
    lora_module_dropout: float = field(
        default=0.0, metadata={"help": "LoRA dropout for disabling module at all."}
    )
    use_effective_conv2d: bool = field(
        default=False, metadata={"help": "Use parameter effective decomposition for Conv2d 3x3 with ksize > 1."}
    )

    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Use gradient checkpointing."}
    )

    save_steps: int = (
        field(default=500, metadata={"help": "Save checkpoint every X updates steps."}),
    )
    save_total_limit: Optional[int] = field(
        default=None, metadata={"help": "Limit the total amount of checkpoints."}
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to folder of checkpoint to resume training from."},
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    data_seed: Optional[int] = field(
        default=None, metadata={"help": "Random seed for data sampler."}
    )

    tracker_init_kwargs: dict = field(
        default_factory=lambda: {}, metadata={"help": "Arguments for tracker init."}
    )

    def __post_init__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_warmup_steps(self, num_training_steps: int):
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else math.ceil(num_training_steps * self.warmup_ratio)
        )

        return warmup_steps

    def get_eval_steps(self, num_training_steps: int):
        eval_steps = (
            int(self.eval_steps)
            if self.eval_steps > 1
            else math.ceil(self.eval_steps * num_training_steps)
        )

        return eval_steps

    def get_project_configuration(self):
        return ProjectConfiguration(
            logging_dir=self.logging_dir,
            total_limit=self.save_total_limit,
        )

    def get_deepspeed_plugin(self):
        return None

    def get_fsdp_plugin(self):
        return None
