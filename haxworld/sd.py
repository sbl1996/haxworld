from typing import Dict, Union, List, Optional, Literal, Tuple
from numbers import Number
import copy

import numpy as np

import jax
import jax.numpy as jnp

from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers import FlaxPNDMScheduler, FlaxDDIMScheduler, FlaxDPMSolverMultistepScheduler

from flax.jax_utils import replicate
from flax.traverse_util import flatten_dict

import torch

from haxworld.pipelines.stable_diffusion import FlaxStableDiffusionPipeline
from haxworld.pipelines.stable_diffusion_img2img import FlaxStableDiffusionImg2ImgPipeline
from haxworld.pipelines.stable_diffusion_xl import FlaxStableDiffusionXLPipeline
from haxworld.conversion import key_to_flax, tensor_to_flax, lora_key_to_hf_param, lora_key_to_alpha


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


@jax.jit
def merge_lora(p, w, down, up, alpha):
    rank = up.shape[1]
    n = len(p.shape)
    if n == 4:
        up = up.transpose(2, 3, 0, 1)
        down = down.transpose(2, 3, 0, 1)
    d = up @ down
    d = jnp.swapaxes(d, -1, -2)
    d = (w * alpha / rank) * d
    return d.astype(p.dtype)


def to_numpy(t):
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().view(jnp.bfloat16)
    else:
        return t.numpy()


def convert_lora_to_flax(state_dict, network_alphas, params):
    params = flatten_dict(params, sep=".")
    lora_dict = {}
    device = None
    for k, v in state_dict.items():
        if ".down." in k:
            param_key = key_to_flax(lora_key_to_hf_param(k))
            assert param_key in params
            p = params[param_key]
            if device is None:
                device = list(p.devices())[0]
            lora_up_key = k.replace(".down.", ".up.")
            alpha = network_alphas[lora_key_to_alpha(k)]
            down = jax.device_put(to_numpy(v), device)
            up = jax.device_put(to_numpy(state_dict[lora_up_key]), device)
            assert jax.eval_shape(merge_lora, p, 0, down, up, alpha).shape == p.shape, f"Mismatch in shape, lora: {k}, param: {param_key}"
            lora_dict[param_key] = (down, up, alpha)
    return lora_dict


class StableDiffusion:

    def __init__(
        self,
        pretrained_model_name_or_path,
        dtype: Union[jax.typing.DTypeLike, str] = jnp.bfloat16,
        scheduler: Optional[Literal["ddim", "pndm", "dpm_solver"]] = None,
        seed: int = 0,
        variant: Literal["sd", "sdxl"] = "sd",
        device = None,
        **kwargs):
        if isinstance(dtype, str):
            dtype = str_to_dtype[dtype]

        if variant == "sd":
            pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, **kwargs)
            upscaler = FlaxStableDiffusionImg2ImgPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                safety_checker=None,
                feature_extractor=pipeline.feature_extractor,
                dtype=pipeline.dtype,                
            )
        elif variant == "sdxl":
            pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
                pretrained_model_name_or_path, split_head_dim=True, **kwargs)
            upscaler = None

        scheduler_state = params.pop("scheduler")
        if scheduler is not None:
            if scheduler == 'dpm_solver':
                scheduler_cls = FlaxDPMSolverMultistepScheduler
            elif scheduler == 'pndm':
                scheduler_cls = FlaxPNDMScheduler
            elif scheduler == 'ddim':
                scheduler_cls = FlaxDDIMScheduler
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")
            scheduler, scheduler_state = scheduler_cls.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")
            pipeline.scheduler = scheduler
            if upscaler is not None:
                upscaler.scheduler = scheduler_cls.from_pretrained(
                    pretrained_model_name_or_path, subfolder="scheduler")[0]

        params = jax.tree.map(lambda x: x.astype(dtype), params)
        scheduler_state = jax.tree.map(lambda x: x.astype(jnp.float32), scheduler_state)
        params["scheduler"] = scheduler_state        

        if device is None:
            device = jax.local_devices()[0]
        params = jax.device_put(params, device)
        
        self.variant = variant
        self.pipeline = pipeline
        self.upscaler = upscaler
        self.params = params
        self.p_params = replicate(params)
        self.num_devices = jax.local_device_count()
        self.seed = seed
        self._gen = np.random.Generator(np.random.PCG64(seed))
        self._current_seed = self._next_seed()

        self._lora = {}
    
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
        height,
        width,
        num_inference_steps=25,
        guidance_scale=7.5,
        negative_prompt=None,
        seed=None,
        upscale: Union[bool, Number, Tuple[Number, Number]] = False,
        upscale_steps: int = 20,
        resize_method: Union[str, jax.image.ResizeMethod] = 'bicubic',
        denoising_strength: float = 0.7,
        antialias: bool = True,
    ):
        do_upscale = False
        if upscale is not False:
            do_upscale = True
            assert self.upscaler is not None
            if upscale is True:
                upscale = 2.0
            if isinstance(upscale, Number):
                upscale_h = upscale_w = upscale
            else:
                upscale_h, upscale_w = upscale

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
            output_type="latent" if do_upscale else "pil",
            jit=True,
        ).images

        if do_upscale:
            images = self.upscaler(
                prompt_ids,
                images,
                self.p_params,
                rng,
                denoising_strength,
                num_inference_steps=int(upscale_steps / denoising_strength) + 1,
                height=int(height * upscale_h),
                width=int(width * upscale_w),
                guidance_scale=guidance_scale,
                neg_prompt_ids=neg_prompt_ids,
                resize_method=resize_method,
                antialias=antialias,
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

    def load_lora_weights(
        self, pretrained_model_name_or_path: str, adapter_name=None, **kwargs):
        if adapter_name is None:
            adapter_name = f"default_{len(self._lora)}"
        if adapter_name in self._lora:
            raise ValueError(f"Adapter {adapter_name} already exists")

        if self.variant == "sdxl" and "unet_config" not in kwargs:
            kwargs["unet_config"] = self.pipeline.unet.config

        state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(
            pretrained_model_name_or_path, **kwargs)
        lora_dict = convert_lora_to_flax(state_dict, network_alphas, self.params)
        self._lora[adapter_name] = {
            "weight": 0.0,
            "params": lora_dict,
        }

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[Union[float, List[float]]] = None,
    ):
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        if adapter_weights is None:
            adapter_weights = 1.0
        if not isinstance(adapter_weights, list):
            adapter_weights = [adapter_weights] * len(adapter_names)

        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}"
            )

        adapter_names = copy.deepcopy(adapter_names)
        adapter_weights = copy.deepcopy(adapter_weights)
        for name in self._lora:
            if name not in adapter_names:
                adapter_names.append(name)
                adapter_weights.append(0.0)

        self.p_params = None
        params = self.params
        
        weights_diff = {}
        for name, weight in zip(adapter_names, adapter_weights):
            weights_diff[name] = weight - self._lora[name]["weight"]
            self._lora[name]["weight"] = weight

        params_t = flatten_dict(params, sep=".")
        for k in list(params_t.keys()):
            p = params_t.pop(k)
            d = 0
            for name in adapter_names:
                lora_p = self._lora[name]["params"].get(k, None)
                w_diff = weights_diff[name]
                if lora_p is not None and w_diff != 0:
                    d += merge_lora(p, w_diff, *lora_p)
            if not isinstance(d, int):
                assign2(params, k, d)

        self.params = params
        self.p_params = replicate(self.params)


def assign2(params, key, diff):
    parts = key.split(".")
    d = params
    for i in range(len(parts) - 1):
        d = d[parts[i]]
    d[parts[-1]] += diff


def assign_inplace(params, key, value):
    parts = key.split(".")
    d = params
    for i in range(len(parts) - 1):
        d = d[parts[i]]
    old_value = d[parts[-1]]
    device = list(old_value.devices())[0]
    d[parts[-1]] = jax.device_put(value, device).astype(old_value.dtype)