import os
from typing import Optional, Literal
from dataclasses import dataclass
import tyro

import torch
from diffusers import StableDiffusionPipeline

@dataclass
class Args:
    source: str
    """Path to the model checkpoint file or directory, can be remote huggingface hub"""
    flax_repo: str
    """Path to the reference flax model repo"""
    output: str
    """Output path to save the dumped flax model"""
    load_dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    """Data type to use for loading the torch model"""
    save_dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    """Data type to use for saving the flax model"""
    vae_path: Optional[str] = None
    """Optional path to the VAE checkpoint file or directory, will overwrite the default VAE"""
    vae_ref: Optional[str] = None
    """Optional path to the reference torch model repo for loading VAE config"""

str_to_dtype = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = tyro.cli(Args)

    load_dtype = str_to_dtype[args.load_dtype]

    pipeline = StableDiffusionPipeline.from_single_file(
        args.source, safety_checker=None,
        torch_dtype=load_dtype,
    )

    if args.vae_path is not None:
        vae = pipeline.vae.__class__.from_single_file(
            args.vae_path,
            torch_dtype=load_dtype,
            config=args.vae_ref,
            subfolder="vae",
        )
        pipeline.vae.load_state_dict(vae.state_dict())
        del vae

    params_sd = {}
    modules = ["text_encoder", "unet", "vae"]
    for m in modules:
        d = getattr(pipeline, m).state_dict()
        for k, v in d.items():
            params_sd[f"{m}.{k}"] = v.numpy()
    del pipeline

    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))
    from haxworld.sd import StableDiffusion
    sd = StableDiffusion(args.flax_repo, dtype=args.save_dtype, safety_checker=None)
    sd.set_params(params_sd)
    sd.pipeline.save_pretrained(args.output, sd.params, safe_serialization=True)