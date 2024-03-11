from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)


def load_pipeline_from_original_sd(ckpt_path: str, **kwargs):
    from_safetensors = ckpt_path.endswith(".safetensors")
    pipeline = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=ckpt_path,
        from_safetensors=from_safetensors,
    )

    return pipeline

def load_pipeline(
    pretrained_name_or_path: str,
    is_from_original_sd: bool = False,
    vae=None,
    text_encoder=None,
    tokenizer=None
):
    override_modules = {}
    if vae is not None:
        override_modules["vae"] = vae
    if text_encoder is not None:
        override_modules["text_encoder"] = text_encoder
    if tokenizer is not None:
        override_modules["tokenizer"] = tokenizer

    if is_from_original_sd:
        pipeline = load_pipeline_from_original_sd(pretrained_name_or_path, **override_modules)
    elif pretrained_name_or_path.endswith((".bin", ".ckpt", ".safetensors")):
        pipeline = StableDiffusionPipeline.from_single_file(pretrained_name_or_path, **override_modules)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_name_or_path,
            **override_modules
        )

    return pipeline
