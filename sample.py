"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import tiktoken
from models.gpt2 import GPTConfig, GPT
from models.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
from models.qwen_tokenizer import Qwen3Tokenizer
from load_critic_4bit import load_critic_model_4bit

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
experiment_name = "pretrain_critic_qwen3_0.6b_base"
use_4bit = False  # whether to load model with 4-bit quantization for memory efficiency
start = "What is the meaning of life?"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 300  # number of tokens generated in each sample
temperature = (
    1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    16  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0+ to compile the model to be faster
model_type = os.environ.get("TOKENIZER", "qwen3").lower()
is_qwen = model_type.startswith("qwen")
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


def get_context_window(model):
    if hasattr(model, "config") and hasattr(model.config, "block_size"):
        return model.config.block_size
    if hasattr(model, "cfg") and isinstance(model.cfg, dict):
        return model.cfg.get("context_length", 1024)
    return 1024


def generate_autoregressive(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Minimal autoregressive generation usable by both GPT and Qwen3 models."""
    context_window = get_context_window(model)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_window else idx[:, -context_window:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    # Construct checkpoint filename with experiment name suffix if available
    checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
    ckpt_path = os.path.join(out_dir, checkpoint_filename)
    
    # If experiment-specific checkpoint doesn't exist, try default
    if not os.path.exists(ckpt_path) and experiment_name:
        default_ckpt_path = os.path.join(out_dir, "ckpt.pt")
        if os.path.exists(default_ckpt_path):
            print(f"Experiment checkpoint '{checkpoint_filename}' not found, using default 'ckpt.pt'")
            ckpt_path = default_ckpt_path
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    if use_4bit:
        print(f"Loading checkpoint with 4-bit quantization from {ckpt_path}")
        model = load_critic_model_4bit(ckpt_path)
        print(f"Model loaded in 4-bit - memory usage reduced by ~75%")
    else:
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Extract only what we need and free the rest
        checkpoint_model_args = checkpoint["model_args"]
        checkpoint_model_type = checkpoint.get(
            "model_type",
            "qwen3" if "context_length" in checkpoint_model_args else "gpt2",
        )
        model_type = checkpoint_model_type
        is_qwen = model_type.startswith("qwen")
        state_dict = checkpoint["model"]
        
        # Free the checkpoint dict immediately - we don't need optimizer states!
        del checkpoint
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        if is_qwen:
            model = Qwen3Model(checkpoint_model_args)
        else:
            gptconf = GPTConfig(**checkpoint_model_args)
            model = GPT(gptconf)
        
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model_type = "gpt2"
    is_qwen = False

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Choose tokenizer based on model type
if is_qwen:
    print("Using Qwen3 tokenizer...")
    repo_id = os.environ.get("QWEN3_TOKENIZER_REPO", "Qwen/Qwen3-0.6B-Base")
    tok_file = os.environ.get("QWEN3_TOKENIZER_FILE", "tokenizer.json")
    qwen_tok = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
    )
    encode = lambda s: qwen_tok.encode(s, chat_wrapped=False)
    decode = lambda l: qwen_tok.decode(l)
else:
    print("Using GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate_autoregressive(
                model,
                x,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            print(decode(y[0].tolist()))
            print("---------------")
