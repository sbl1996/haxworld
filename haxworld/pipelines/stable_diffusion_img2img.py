# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from functools import partial
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel

from diffusers.models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from diffusers.schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, logging, replace_example_docstring
from diffusers.pipelines.pipeline_flax_utils import FlaxDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import FlaxStableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import jax
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from flax.jax_utils import replicate
        >>> from flax.training.common_utils import shard
        >>> import requests
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from diffusers import FlaxStableDiffusionImg2ImgPipeline


        >>> def create_key(seed=0):
        ...     return jax.random.PRNGKey(seed)


        >>> rng = create_key(0)

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> response = requests.get(url)
        >>> init_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> init_img = init_img.resize((768, 512))

        >>> prompts = "A fantasy landscape, trending on artstation"

        >>> pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4",
        ...     revision="flax",
        ...     dtype=jnp.bfloat16,
        ... )

        >>> num_samples = jax.device_count()
        >>> rng = jax.random.split(rng, jax.device_count())
        >>> prompt_ids, processed_image = pipeline.prepare_inputs(
        ...     prompt=[prompts] * num_samples, image=[init_img] * num_samples
        ... )
        >>> p_params = replicate(params)
        >>> prompt_ids = shard(prompt_ids)
        >>> processed_image = shard(processed_image)

        >>> output = pipeline(
        ...     prompt_ids=prompt_ids,
        ...     image=processed_image,
        ...     params=p_params,
        ...     prng_seed=rng,
        ...     strength=0.75,
        ...     num_inference_steps=50,
        ...     jit=True,
        ...     height=512,
        ...     width=768,
        ... ).images

        >>> output_images = pipeline.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
        ```
"""


class FlaxStableDiffusionImg2ImgPipeline(FlaxDiffusionPipeline):
    r"""
    Flax-based pipeline for text-guided image-to-image generation using Stable Diffusion.

    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.FlaxCLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`FlaxUNet2DConditionModel`]):
            A `FlaxUNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`], or
            [`FlaxDPMSolverMultistepScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_inputs(self, prompt: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")

        if isinstance(image, Image.Image):
            image = [image]

        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids, processed_images

    def get_timestep_start(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        return t_start

    def _get_embeddings(self, prompt_ids: jnp.ndarray, params):
        return self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

    def get_embeddings(self, prompt_ids: jnp.array, params: Union[Dict, FrozenDict], jit=True):
        if jit:
            func = partial(_p_get_embeddings, pipe=self)
        else:
            func = self._get_embeddings
        max_length = self.tokenizer.model_max_length
        length = prompt_ids.shape[-1]
        if length > max_length:
            assert length % max_length == 0
            chunks = []
            for i in range(length // max_length):
                chunk = prompt_ids[..., i * max_length : (i + 1) * max_length]
                chunks.append(func(chunk, params))
            return jnp.concatenate(chunks, axis=-2)
        return func(prompt_ids, params)

    def _generate(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        start_timestep: int,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        noise: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
        prompt_embeds: Optional[jnp.ndarray] = None,
        neg_prompt_embeds: Optional[jnp.ndarray] = None,
        resize_method: Union[str, jax.image.ResizeMethod] = 'bicubic',
        antialias: bool = True,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if prompt_embeds is None:
            prompt_embeds = self._get_embeddings(prompt_ids, params)

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_embeds.shape[0]

        if neg_prompt_embeds is None:
            if neg_prompt_ids is None:
                max_length = prompt_embeds.shape[-1]
                uncond_input = self.tokenizer(
                    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
                ).input_ids
            else:
                uncond_input = neg_prompt_ids
            neg_prompt_embeds = self._get_embeddings(uncond_input, params)
        context = jnp.concatenate([neg_prompt_embeds, prompt_embeds])

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if noise is None:
            noise = jax.random.normal(prng_seed, shape=latents_shape, dtype=jnp.float32)
        else:
            if noise.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {noise.shape}, expected {latents_shape}")

        if image.shape[1] == self.unet.config.in_channels:
            init_latents = image
        else:
            init_latent_dist = self.vae.apply({"params": params["vae"]}, image, method=self.vae.encode).latent_dist
            init_latents = init_latent_dist.sample(key=prng_seed).transpose((0, 3, 1, 2))
            init_latents = self.vae.config.scaling_factor * init_latents

        target_h = height // self.vae_scale_factor
        target_w = width // self.vae_scale_factor
        init_shape = init_latents.shape
        if init_shape[2] != target_h or init_shape[3] != target_w:
            init_latents = jax.image.resize(
                init_latents, (*init_shape[:2], target_h, target_w),
                method=resize_method, antialias=antialias,
            )

        def loop_body(step, args):
            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = self.scheduler.scale_model_input(scheduler_state, latents_input, t)

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return latents, scheduler_state

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents_shape
        )

        latent_timestep = scheduler_state.timesteps[start_timestep : start_timestep + 1].repeat(batch_size)

        latents = self.scheduler.add_noise(params["scheduler"], init_latents, noise, latent_timestep)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        if DEBUG:
            # run with python for loop
            for i in range(start_timestep, num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(start_timestep, num_inference_steps, loop_body, (latents, scheduler_state))

        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_ids: jnp.ndarray,
        image: jnp.ndarray,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jnp.ndarray] = 7.5,
        noise: jnp.ndarray = None,
        neg_prompt_ids: jnp.ndarray = None,
        prompt_embeds: Optional[jnp.ndarray] = None,
        neg_prompt_embeds: Optional[jnp.ndarray] = None,
        resize_method: Union[str, jax.image.ResizeMethod] = 'bicubic',
        antialias: bool = True,
        return_dict: bool = True,
        jit: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt_ids (`jnp.ndarray`):
                The prompt or prompts to guide image generation.
            image (`jnp.ndarray`):
                Array representing an image batch to be used as the starting point.
            params (`Dict` or `FrozenDict`):
                Dictionary containing the model parameters/weights.
            prng_seed (`jax.Array` or `jax.Array`):
                Array containing random number generator key.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            noise (`jnp.ndarray`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. The array is generated by
                sampling using the supplied random `generator`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions.

                    <Tip warning={true}>

                    This argument exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a
                    future release.

                    </Tip>

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated images
                and the second element is a list of `bool`s indicating whether the corresponding generated image
                contains "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]

        start_timestep = self.get_timestep_start(num_inference_steps, strength)

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                image,
                params,
                prng_seed,
                start_timestep,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                noise,
                neg_prompt_ids,
                prompt_embeds,
                neg_prompt_embeds,
                resize_method,
                antialias,
            )
        else:
            images = self._generate(
                prompt_ids,
                image,
                params,
                prng_seed,
                start_timestep,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                noise,
                neg_prompt_ids,
                prompt_embeds,
                neg_prompt_embeds,
                resize_method,
                antialias,
            )

        images = np.asarray(images)
        has_nsfw_concept = False

        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


# Static argnums are pipe, start_timestep, num_inference_steps, height, width. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, None, None, None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 5, 6, 7, 8, 14, 15),
)
def _p_generate(
    pipe,
    prompt_ids,
    image,
    params,
    prng_seed,
    start_timestep,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    noise,
    neg_prompt_ids,
    prompt_embeds,
    neg_prompt_embeds,
    resize_method,
    antialias,
):
    return pipe._generate(
        prompt_ids,
        image,
        params,
        prng_seed,
        start_timestep,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        noise,
        neg_prompt_ids,
        prompt_embeds,
        neg_prompt_embeds,
        resize_method,
        antialias,
    )


@partial(
    jax.pmap,
    in_axes=(None, 0, 0),
    static_broadcasted_argnums=(0,),
)
def _p_get_embeddings(
    pipe,
    prompt_ids,
    params,
):
    return pipe._get_embeddings(
        prompt_ids,
        params,
    )


def preprocess(image, dtype):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return 2.0 * image - 1.0
