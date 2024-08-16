from typing import Optional, Literal
from dataclasses import dataclass
import tyro


import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.loaders import StableDiffusionLoraLoaderMixin

import torch

from haxworld.pipelines.stable_diffusion import FlaxStableDiffusionPipeline


@dataclass
class Args:
    source: str
    """Path to the model file or directory, can be remote huggingface hub"""
    output: str
    """Output filename (without suffix) to save the dumped output"""
    type: Literal["flax", "checkpoint", "lora"]
    """Type of the model"""
    variant: Literal["sd", "sdxl"] = "sd"
    """Variant of the model"""
    weight_name: Optional[str] = None
    """Name of the lora weight file"""
    shape: bool = True
    """Print shape of the parameters"""

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.variant == 'sd':
        modules = ["text_encoder", "unet", "vae"]
    else:
        modules = ["text_encoder", "text_encoder_2", "unet", "vae"]

    if args.type == 'flax':
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            args.source,
            dtype=jnp.bfloat16,
            safety_checker=None,
            revision="bf16",
        )

        lines = []
        for m in modules:
            d = flatten_dict(params[m], sep=".")
            for k in sorted(d.keys()):
                line = f"{m}.{k}"
                if args.shape:
                    line += f" {tuple(d[k].shape)}"
                lines.append(line)
        with open(f"{args.output}.txt", "w") as f:
            f.write("\n".join(lines))
    elif args.type == 'checkpoint':
        pipeline_cls = {
            "sd": StableDiffusionPipeline,
            "sdxl": StableDiffusionXLPipeline,
        }[args.variant]
        pipeline = pipeline_cls.from_single_file(
            args.source,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

        lines = []
        for m in modules:
            d = getattr(pipeline, m).state_dict()
            for k in sorted(d.keys()):
                line = f"{m}.{k}"
                if args.shape:
                    line += f" {tuple(d[k].shape)}"
                lines.append(line)
        with open(f"{args.output}.txt", "w") as f:
            f.write("\n".join(lines))
    else:
        kwargs = {}
        if args.weight_name is not None:
            kwargs["weight_name"] = args.weight_name
        state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(
            args.source, **kwargs)
        
        lines = []
        d = state_dict
        for k in sorted(d.keys()):
                line = k
                if args.shape:
                    line += f" {tuple(d[k].shape)}"
                lines.append(line)
        with open(f"{args.output}.txt", "w") as f:
            f.write("\n".join(lines))

        lines = []
        for k in sorted(network_alphas.keys()):
            lines.append(f"{k} (1,)")
        with open(f"{args.output}_alpha.txt", "w") as f:
            f.write("\n".join(lines))