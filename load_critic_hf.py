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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype or _default_dtype()

    print(f"Loading HF critic model '{model_id}' on {device} (dtype={dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

