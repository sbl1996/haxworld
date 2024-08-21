import os
import time
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np

import jax.numpy as jnp
from haxworld.sd import StableDiffusion


def load_sd(model_path, lora_dir=None):
    sd = StableDiffusion(
        model_path,
        dtype=jnp.bfloat16,
        scheduler='dpm_solver',
        safety_checker=None
    )
    if lora_dir:
        sd.add_lora_dir(lora_dir)
    return sd


def load_model(state, model, height, width):
    if model is None:
        return model, height, width
    models = state['models']
    start = time.time()
    model_path = models[model]
    if model_path != state['model_path']:
        sd = load_sd(model_path, state['lora_dir'])
        state['sd'] = sd
        state['model'] = model
        state['model_path'] = model_path
        if sd.variant == 'sd':
            height, width = 512, 512
        elif sd.variant == 'sdxl':
            height, width = 1024, 1024
        print(f"Loaded in {time.time()-start:.1f} seconds")
    return model, height, width


def generate_images(
    state, prompt, neg_prompt, sample_steps, sampler, batch_size, batch_count,
    height, width, resize_method, upscale, denoising_strength, cfg_scale,
    seed, model):
    sd: StableDiffusion = state['sd']
    if sd is None:
        return f"Error: No model loaded", []

    seed_max = 2**32-1
    seed = int(seed)
    if seed == -1:
        gen_seed = int(np.random.randint(0, seed_max))
    else:
        gen_seed = seed
    gen = np.random.Generator(np.random.PCG64(gen_seed))

    antialias = False
    if resize_method is not None and "antialias" in resize_method:
        resize_method_ = resize_method.replace(" (antialiased)", "")
        antialias = True
    else:
        resize_method_ = resize_method
    try:
        all_images = []
        for i in range(batch_count):
            print(f"Generating image {i+1}/{batch_count}")
            seed = gen.integers(0, seed_max)
            images = sd.generate(
                prompt, negative_prompt=neg_prompt,
                height=height, width=width,
                num_inference_steps=sample_steps,
                batch_size=batch_size // 4,
                upscale=upscale,
                resize_method=resize_method_,
                antialias=antialias,
                denoising_strength=denoising_strength,
                guidance_scale=cfg_scale,
                seed=seed,
                long_prompt=False,
            )
            for image in images:
                save_dir = state['save_dir']
                save_path = save_dir / f"{state['n']}.jpg"
                image.save(save_path)
                state['n'] += 1
                all_images.append(image)

        response_text = \
            f"Prompt: {prompt}\n\nNegative prompt: {neg_prompt}\n\n" \
            f"Model: {model}, Resolution: {width}x{height}, CFG scale: {cfg_scale}, Seed: {gen_seed}\n" \
            f"Steps: {sample_steps}, Sampler: {sampler}, Batch size: {batch_size}, Batch count: {batch_count}\n" \
            f"Upscale: {upscale}, Resize: {resize_method}, Denoising strength: {denoising_strength}\n"
        return response_text, all_images
    except Exception as e:
        return f"Error: {str(e)}", []


custom_css = """
    #prompt {
        display: flex;
        flex-direction: row;
    }
    #params {
        display: flex;
        flex-direction: row;
    }
    .my-flex-row {
        flex-direction: row;
    }
"""

def set_dir(state, k, d):
    d = Path(d).expanduser().absolute()
    state[k] = d
    return d

def set_save_dir(state, d):
    d = set_dir(state, 'save_dir', d)
    if not d.exists():
        d.mkdir(parents=True)


def list_lora_dir(d):
    res = "\n".join([ f.stem for f in Path(d).glob("*.safetensors") ])
    return res, res


def update_models(state, model_dir):
    models = {}
    for d in model_dir.iterdir():
        if d.is_dir():
            models[d.name] = str(d)
    state['models'] = models
    return list(models.keys())


def update_model_dir(state, model_dir):
    set_dir(state, 'model_dir', model_dir)
    models = update_models(state, state['model_dir'])
    return "\n".join(models), gr.update(choices=models, value=None)


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    from jax_smi import initialise_tracking
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))
    initialise_tracking()

    root = os.getenv("HW_ROOT", ".")
    default_model_dir = os.path.join(root, "model")
    default_lora_dir = os.path.join(root, "lora")
    default_save_dir = os.path.join(root, "results")

    state = {
        'n': 0, 'model': None, 'model_path': None, 'sd': None
    }
    set_dir(state, 'lora_dir', default_lora_dir)
    set_dir(state, 'model_dir', default_model_dir)
    set_save_dir(state, default_save_dir)

    update_models(state, state['model_dir'])
    loras = list_lora_dir(state['lora_dir'])[0]

    with gr.Blocks(css=custom_css) as app:
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row(elem_id="prompt"):
                        with gr.Column(elem_classes="my-flex-row", scale=3):
                            prompt = gr.Textbox(label="Prompt", show_label=False, lines=4, placeholder="Prompt")
                            neg_prompt = gr.Textbox(label="Negative prompt", show_label=False, lines=4, placeholder="Negative prompt")
                        with gr.Column(elem_classes="my-flex-row", scale=1, min_width=0):
                            model = gr.Dropdown(list(state['models'].keys()), label="Model")
                            lora_dir_files_0 = gr.Textbox(autoscroll=False, label="LoRA", value=loras)
                    with gr.Row(elem_id="params"):
                        with gr.Column(elem_classes="my-flex-row", scale=1, min_width=0):
                            width = gr.Slider(minimum=64, maximum=2048, step=32, label="Width", value=512, elem_id="txt2img_width")
                            height = gr.Slider(minimum=64, maximum=2048, step=32, label="Height", value=512, elem_id="txt2img_height")
                        with gr.Column(elem_classes="my-flex-row", scale=1, min_width=0):
                            sampler = gr.Dropdown(["ddim", "pndm", "dpm_solver"], label="Sampler", value="dpm_solver")
                            sample_steps = gr.Slider(minimum=1, maximum=100, step=1, label='Sample steps', value=30, elem_id="txt2img_steps")
                        with gr.Column(elem_classes="my-flex-row", scale=1, min_width=0):
                            batch_count = gr.Slider(minimum=1, maximum=8, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                            batch_size = gr.Slider(minimum=4, maximum=32, step=4, label='Batch size', value=4, elem_id="txt2img_batch_size")
                    with gr.Row():
                        resize_methods = ["nearest", "linear", "bicubic", "lanczos3", "lanczos5"]
                        resize_methods_ = []
                        for m in resize_methods:
                            resize_methods_.append(m)
                            resize_methods_.append(m + " (antialiased)")
                        upscale = gr.Slider(minimum=1, maximum=4, step=0.25, label='Upscale by', value=1, elem_id="txt2img_upscale")
                        resize_method = gr.Dropdown(resize_methods_, label="Upscaler", value=resize_methods_[5])
                        denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='Denoising strength', value=0.75)
                    with gr.Row():
                        cfg_scale = gr.Slider(minimum=1, maximum=15, step=0.5, label='CFG scale', value=7, elem_id="txt2img_cfg_scale")
                        seed = gr.Textbox(label="Seed", value="-1", show_copy_button=True)
                with gr.Column(scale=2):
                    with gr.Row():
                        generate_button = gr.Button("Generate")
                    with gr.Row():
                        images_output = gr.Gallery(label="Image")
                    with gr.Row():
                        result_text = gr.Textbox(label="Output", lines=4, interactive=False)
        with gr.Tab("setting"):
            with gr.Column():
                with gr.Row(variant='compact'):
                    with gr.Column(min_width=0):
                        model_dir = gr.Textbox(label="Model directory", value=str(state['model_dir']))
                        btn_update_model_dir = gr.Button(value="Update")
                    with gr.Column(min_width=0):
                        model_dir_children = gr.Textbox(lines=4, max_lines=4, autoscroll=False, label="Models")
                    btn_update_model_dir.click(
                        partial(update_model_dir, state),
                        model_dir,
                        [model_dir_children, model]
                    )
                with gr.Row(variant='compact'):
                    with gr.Column(min_width=0):
                        lora_dir = gr.Textbox(label="LoRA directory", value=str(state['lora_dir']))
                        btn_update_lora_dir = gr.Button(value="Update")
                    with gr.Column(min_width=0):
                        lora_dir_files = gr.Textbox(lines=4, max_lines=4, autoscroll=False, label="LoRA list")
                    btn_update_lora_dir.click(list_lora_dir, lora_dir, [lora_dir_files, lora_dir_files_0])
                save_dir = gr.Textbox(label="Save directory", value=str(state['save_dir']))

        generate_button.click(
            partial(generate_images, state),
            inputs=[
                prompt, neg_prompt, sample_steps, sampler, batch_size, batch_count,
                height, width, resize_method, upscale, denoising_strength, cfg_scale,
                seed, model],
            outputs=[result_text, images_output]
        )
        model.change(
            fn=partial(load_model, state),
            inputs=[model, height, width],
            outputs=[model, height, width],
        )
        save_dir.change(
            fn=partial(set_save_dir, state),
            inputs=[save_dir],
            outputs=[save_dir],
        )

    app.launch(share=False)