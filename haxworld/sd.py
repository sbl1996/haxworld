from typing import Dict, Union

import numpy as np

import jax
import jax.numpy as jnp

from flax.jax_utils import replicate
from flax.traverse_util import flatten_dict
from haxworld.pipelines.stable_diffusion import FlaxStableDiffusionPipeline

_conversion_map = [
    ("to_q", "query"),
    ("to_k", "key"),
    ("to_v", "value"),
    ("to_out_0", "proj_attn"),
]

def key_to_flax(k):
    if "embeddings" in k:
        k = k.replace(".weight", ".embedding")
    elif "norm" in k:
        k = k.replace(".weight", ".scale")
    if not k.startswith("text_encoder"):
        for i in range(10):
            k = k.replace(f".{i}.", f"_{i}.")
    if k.startswith("vae"):
        for s1, s2 in _conversion_map:
            k = k.replace(s1, s2)
    return k.replace(".weight", ".kernel")


def tensor_to_flax(k, v):
    if k.endswith("kernel"):
        shape = v.shape
        if len(shape) == 2:
            v = v.T
        elif len(shape) == 4:
            v = v.transpose(2, 3, 1, 0)
    return v


# def tensor_to_flax(k, shape):
#     if k.endswith("kernel"):
#         if len(shape) == 2:
#             shape = (shape[1], shape[0])
#         elif len(shape) == 4:
#             shape = (shape[2], shape[3], shape[1], shape[0])
#     return shape
def lora_key_to_hf_param(k):
    if k.startswith("text_encoder"):
        name1 = "lora_linear_layer"
        name2 = "_lora"
        if name1 in k:
            k = k.replace(f"{name1}.down")
            k = k.replace(f"{name1}.up")
        elif name2 in k:
            pass
    return k


def tokenize_prompt(pipeline, prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    if neg_prompt is not None:
        neg_prompt = ""
        neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    else:
        neg_prompt_ids = None
    return prompt_ids, neg_prompt_ids


def replicate_all(prompt_ids, neg_prompt_ids, rng, num_devices):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.split(rng, num_devices)
    return p_prompt_ids, p_neg_prompt_ids, rng


str_to_dtype = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


class StableDiffusion:

    def __init__(
        self,
        pretrained_model_name_or_path,
        dtype: Union[jax.typing.DTypeLike, str] = jnp.bfloat16,
        seed: int = 0,
        device = None,
        **kwargs):
        if isinstance(dtype, str):
            dtype = str_to_dtype[dtype]

        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            dtype=dtype,
            **kwargs,
        )
        
        if device is None:
            device = jax.local_devices()[0]
        params = jax.device_put(params, device)
        
        self.pipeline = pipeline
        self.params = params
        self.p_params = replicate(params)
        self.num_devices = jax.local_device_count()
        self.seed = seed
        self._gen = np.random.Generator(np.random.PCG64(seed))
        self._current_seed = self._next_seed()
    
    def _next_seed(self):
        return self._gen.integers(2**30, dtype=np.uint32)
    
    def get_current_seed(self):
        return self._current_seed

    def _get_rng(self, seed=None):
        if seed is None:
            self._current_seed = self._next_seed()
            seed = self._current_seed
        return jax.random.PRNGKey(seed)
    
    def compile(self):
        default_prompt = "high-quality photo of a baby dolphin playing in a pool and wearing a party hat"
        default_neg_prompt = "illustration, low-quality"
        default_seed = 33
        default_guidance_scale = 5.0
        default_num_steps = 25
        self.generate(
            default_prompt,
            num_inference_steps=default_num_steps,
            guidance_scale=default_guidance_scale,
            negative_prompt=default_neg_prompt,
            seed=default_seed,
        )

    def generate(
        self,
        prompt,
        height=None,
        width=None,
        num_inference_steps=25,
        guidance_scale=7.5,
        negative_prompt=None,
        seed=None):
        prompt_ids, neg_prompt_ids = tokenize_prompt(self.pipeline, prompt, negative_prompt)
        rng = self._get_rng(seed)
        prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, rng, self.num_devices)
        images = self.pipeline(
            prompt_ids,
            self.p_params,
            rng,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            neg_prompt_ids=neg_prompt_ids,
            guidance_scale=guidance_scale,
            jit=True,
        ).images

        # convert the images to PIL
        images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
        images = self.pipeline.numpy_to_pil(np.array(images))
        return images

    def _set_params(self, params, new_params):
        keys = list(flatten_dict(params, sep=".").keys())
        should_convert = False
        for k in new_params.keys():
            if k not in keys:
                should_convert = True
                break
        for k, v in new_params.items():
            if should_convert:
                k = key_to_flax(k)
                v = tensor_to_flax(k, v)
            assign_inplace(params, k, v)
        return params

    def set_params(self, params: Dict[str, np.ndarray]):
        self.p_params = None
        self.params = self._set_params(self.params, params)
        self.p_params = replicate(self.params)


def assign_inplace(params, key, value):
    parts = key.split(".")
    d = params
    for i in range(len(parts) - 1):
        d = d[parts[i]]
    old_value = d[parts[-1]]
    device = list(old_value.devices())[0]
    d[parts[-1]] = jax.device_put(value, device).astype(old_value.dtype)