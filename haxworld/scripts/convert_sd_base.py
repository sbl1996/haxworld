import os
from typing import Literal
from dataclasses import dataclass
import tyro

@dataclass
class Args:
    source: str
    """Path to the model checkpoint file or directory, can be remote huggingface hub"""
    output: str
    """Output path to save the dumped flax model"""
    save_dtype: Literal["fp32", "fp16", "bf16"] = "bf16"
    """Data type to use for saving the flax model"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))
    from haxworld.sd import StableDiffusion
    sd = StableDiffusion(args.source, dtype=args.save_dtype, safety_checker=None)
    sd.pipeline.save_pretrained(args.output, sd.params, safe_serialization=True)