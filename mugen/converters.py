from typing import Dict
import torch

from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import save_file

from .loaders import load_pipeline_from_original_sd


def original_sd_to_diffusers(ckpt_path: str, dump_path: str, **kwargs):
    pipeline = load_pipeline_from_original_sd(ckpt_path, **kwargs)
    pipeline.save_pretrained(dump_path)


LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_ADAPTER_NAME = "default"


def get_module_kohya_state_dict(
    module: PeftModel,
    prefix: str,
    dtype: torch.dtype,
    peft_prefix: str = "base_model.model",
    adapter_name: str = LORA_ADAPTER_NAME,
) -> Dict[str, torch.Tensor]:
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        if peft_key.startswith(peft_prefix):
            kohya_key = peft_key.replace(peft_prefix, prefix)
            kohya_key = kohya_key.replace("lora_A", "lora_down")
            kohya_key = kohya_key.replace("lora_B", "lora_up")
            kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
            kohya_ss_state_dict[kohya_key] = weight.to(dtype)

            # Set alpha parameter
            if "lora_down" in kohya_key:
                alpha_key = f'{kohya_key.split(".")[0]}.alpha'
                kohya_ss_state_dict[alpha_key] = torch.tensor(
                    module.peft_config[adapter_name].lora_alpha
                ).to(dtype)

    return kohya_ss_state_dict


def peft_model_to_adapter(peft_model: PeftModel, dump_path: str):
    kohya_ss_state_dict = {}
    dtype = torch.float32

    kohya_ss_state_dict.update(
        get_module_kohya_state_dict(
            peft_model,
            LORA_PREFIX_UNET,
            dtype,
            peft_prefix="base_model.model.unet",
        )
    )
    kohya_ss_state_dict.update(
        get_module_kohya_state_dict(
            peft_model,
            LORA_PREFIX_TEXT_ENCODER,
            dtype,
            peft_prefix="base_model.model.text_encoder",
        )
    )

    save_file(kohya_ss_state_dict, dump_path)
