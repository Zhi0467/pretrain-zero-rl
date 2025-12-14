import os
import torch
from transformers import AutoModelForCausalLM


def _default_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_critic_model_hf(model_id: str, device: str | None = None, dtype=None):
    """
    Lightweight HF critic loader used by avatarl.py
    - Loads a causal LM (teacher) from HuggingFace
    - Puts it in eval mode and disables gradients
    - Moves to the requested device
    """
    if device is None:
        if torch.cuda.is_available():
            # Respect LOCAL_RANK so each DDP rank loads to its own GPU
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    dtype = dtype or _default_dtype()
    # Avoid HF tensor-parallel sharding (triggered with device_map="auto" under DDP)
    # that can try to wrap non-floating shards in Parameters and crash.
    device_map = {"": device}

    print(f"Loading HF critic model '{model_id}' on {device} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

