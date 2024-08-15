
_vae_map = [
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
        for s1, s2 in _vae_map:
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


def tensor_shape_to_flax(k, shape):
    if k.endswith("kernel"):
        if len(shape) == 2:
            shape = (shape[1], shape[0])
        elif len(shape) == 4:
            shape = (shape[2], shape[3], shape[1], shape[0])
    return shape


_lora_text_encoder_map = [
    ("lora_linear_layer.", ""),
    ("to_q_lora", "q_proj"),
    ("to_k_lora", "k_proj"),
    ("to_v_lora", "v_proj"),
    ("to_out_lora", "out_proj"),
]


_lora_unet_map = [
    (".lora.", "."),
    ("to_q_lora", "to_q"),
    ("to_k_lora", "to_k"),
    ("to_v_lora", "to_v"),
    ("to_out_lora", "to_out.0"),
]

def lora_key_to_hf_param(k):
    if k.startswith("text_encoder"):
        for s1, s2 in _lora_text_encoder_map:
            if s1 in k:
                k = k.replace(s1, s2)
                k = k.replace("up.", "")
                k = k.replace("down.", "")
                break
    elif k.startswith("unet"):
        k = k.replace(".processor", "")
        for s1, s2 in _lora_unet_map:
            if s1 in k:
                k = k.replace(s1, s2)
                k = k.replace("up.", "")
                k = k.replace("down.", "")
                break
    else:
        raise ValueError(f"Unknown key {k}")
    return k


def lora_alpha_to_lora_key(k):
    k = k[:-6]
    if "lora" not in k:
        k = k + ".lora.down.weight"
    return k


def lora_key_to_alpha(k):
    if ".lora." in k:
        k = k[:-len(".lora.down.weight")]
    k = k + ".alpha"
    return k