"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from tqdm import tqdm

from models.gpt2 import GPTConfig, GPT
from load_critic_hf import load_critic_model_hf
from models.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

# Import all configuration variables
from config.train_avatarl import *
# load wandb api key, project, and entity from .env
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import wandb
# Load .env (prefer repo root, but also respect any parent/cwd .env)
repo_root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(find_dotenv())  # pick up a .env in cwd/parents if present
load_dotenv(dotenv_path=repo_root_env)  # ensure repo-root .env is loaded
wandb_api_key = os.environ.get("WANDB_API_KEY")
wandb_project = os.environ.get("WANDB_PROJECT")
if not wandb_api_key or not wandb_project:
    raise ValueError("WANDB_API_KEY and WANDB_PROJECT must be set")
wandb.login(key=wandb_api_key)

# -----------------------------------------------------------------------------
# Create config dictionary for logging
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# AvataRL helper functions
# -----------------------------------------------------------------------------

def _get_tokenizer_name() -> str:
    # start.sh exports TOKENIZER; default to gpt2 for backward compatibility
    return os.environ.get("TOKENIZER", "qwen3").lower()


def _get_vocab_size_for_tokenizer(tokenizer_name: str) -> int:
    if tokenizer_name == "qwen3":
        return int(QWEN_CONFIG_06_B["vocab_size"])
    return 50304  # GPT-2 vocab_size, padded for efficiency


def _get_bin_dtype_for_vocab_size(vocab_size: int):
    # Qwen3 vocab doesn't fit in uint16
    return np.uint32 if vocab_size > np.iinfo(np.uint16).max else np.uint16


def _configure_optimizers_generic(model, weight_decay, learning_rate, betas, device_type):
    """
    Generic AdamW optimizer configuration (GPT-style):
    - Apply weight decay to 2D+ parameters; none to biases/norms (1D)
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")
    return optimizer



def load_critic_model(checkpoint_path: str):
    """
    Load a pre-trained critic model.
    Supports:
      - Local GPT-based checkpoints (FP16 or 4-bit)
      - HF models via CRITIC_HF_MODEL env or `hf:` prefix on checkpoint_path
    """
    # env overrides
    env_hf_model = os.environ.get("CRITIC_HF_MODEL")
    env_use_4bit = os.environ.get("USE_4BIT_CRITIC")
    effective_use_4bit = use_4bit_critic if env_use_4bit is None else env_use_4bit.lower() in {"1", "true", "yes"}

    # HF path support
    if env_hf_model or checkpoint_path.startswith("hf:"):
        hf_model_id = env_hf_model or checkpoint_path[len("hf:"):]
        print(f"Loading critic from HuggingFace: {hf_model_id}")
        critic_model = load_critic_model_hf(hf_model_id)
        critic_model.eval()
        for param in critic_model.parameters():
            param.requires_grad = False
        return critic_model

    # Check if we should use 4-bit quantization (from config or env)
    if effective_use_4bit:
        from load_critic_4bit import load_critic_model_4bit
        print(f"Loading critic with 4-bit quantization for ~75% memory reduction")
        return load_critic_model_4bit(checkpoint_path)

    # Original FP16 path
    print(f"Loading critic model in FP16 from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    checkpoint_model_args = checkpoint["model_args"]
    state_dict = checkpoint["model"]

    # Free the checkpoint dict immediately - we don't need optimizer states!
    del checkpoint
    torch.cuda.empty_cache()

    # Create model with checkpoint config
    gptconf = GPTConfig(**checkpoint_model_args)
    critic_model = GPT(gptconf)

    # Remove unwanted prefix if present
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    critic_model.load_state_dict(state_dict)
    critic_model.cuda()
    critic_model.eval()  # Set to eval mode

    # Disable gradients for critic
    for param in critic_model.parameters():
        param.requires_grad = False

    print(f"Critic loaded in FP16 - {checkpoint_model_args['n_layer']} layers, {checkpoint_model_args['n_embd']} dim")
    print(f"Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    return critic_model


def _extract_logits(output):
    """Normalize logits extraction across GPT and HF models."""
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Unsupported model output type for logits extraction: {type(output)}")


def compute_avatarl_loss(
    student_logits: Tensor, 
    critic_logits: Tensor, 
    ground_truth_tokens: Tensor,
    reality_weight: float = 0.7,
    mentor_weight: float = 0.3,
    label_smoothing_epsilon: float = 0.1,
    reward_scale: float = 100.0,
    top_k: int = 16,
    entropy_coefficient: float = 0.01,
    max_reward_clamp: float = None
) -> Tensor:
    """
    Compute the AvataRL policy gradient loss with active token label smoothing.
    
    This implements a Product of Experts (PoE) reward model that combines:
    1. Reality Expert: Active token label-smoothed distribution
       - 90% probability to ground truth token
       - 10% spread ONLY across active tokens (student top-k + critic top-k)
       - Unlike standard label smoothing that spreads across all vocab_size tokens,
         this concentrates the smoothing mass on the ~33 tokens that actually matter
    2. Mentor Expert: Teacher model's distribution over plausible tokens
    
    The key innovation is that label smoothing epsilon is distributed only among
    tokens in the action space (2*top_k + 1 unique tokens), not wasted on the
    entire vocabulary. This provides stronger exploration signal for relevant alternatives.
    
    Args:
        student_logits: Student model's output logits. Shape: (batch_size, seq_len, vocab_size)
        critic_logits: Teacher model's output logits. Shape: (batch_size, seq_len, vocab_size)
        ground_truth_tokens: Ground-truth target tokens. Shape: (batch_size, seq_len)
        reality_weight: Weight for reality expert in PoE (default: 0.7)
        mentor_weight: Weight for mentor expert in PoE (default: 0.3)
        label_smoothing_epsilon: Label smoothing parameter distributed over active tokens only (default: 0.1)
        reward_scale: Scale factor for rewards (default: 100.0)
        top_k: Number of top tokens to consider from both student and critic (default: 16)
        entropy_coefficient: Coefficient for entropy regularization (default: 0.01)
        max_reward_clamp: Maximum reward value to prevent gradient explosion (default: None, uses global config)
        
    Returns:
        tuple: (loss, metrics_dict)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Validate input shapes
    assert student_logits.shape == critic_logits.shape, \
        f"Student and critic logits shape mismatch: {student_logits.shape} vs {critic_logits.shape}"
    assert ground_truth_tokens.shape == (batch_size, seq_len), \
        f"Ground truth shape mismatch: expected {(batch_size, seq_len)}, got {ground_truth_tokens.shape}"
    
    # Reshape to (batch_size * seq_len, vocab_size) for easier processing
    student_logits_flat = student_logits.view(-1, vocab_size)
    critic_logits_flat = critic_logits.view(-1, vocab_size)
    ground_truth_flat = ground_truth_tokens.view(-1)
    
    # --- Step 1: Define Expanded Action Space (Student + Teacher + Ground Truth) ---
    # Get student's top-k predictions
    _, student_top_k_indices = student_logits_flat.topk(top_k, dim=-1)
    
    # Combine student top-k and ground truth indices (critic actions excluded)
    # Shape: (batch_size * seq_len, top_k + 1)
    combined_indices = torch.cat([
        student_top_k_indices,
        ground_truth_flat.unsqueeze(1)
    ], dim=1)
    
    # FULLY VECTORIZED: Remove duplicates while keeping first occurrence
    # No python loops - pure GPU tensor operations
    batch_size_seq = combined_indices.size(0)
    max_actions = top_k + 1  # student top-k + ground truth
    
    # Sort each row and find unique consecutive values
    sorted_indices, sort_idx = combined_indices.sort(dim=1)
    
    # Create mask for unique values (mark first occurrence of each unique value)
    unique_mask = torch.cat([
        torch.ones(batch_size_seq, 1, dtype=torch.bool, device=sorted_indices.device),
        sorted_indices[:, 1:] != sorted_indices[:, :-1]
    ], dim=1)
    
    # To preserve original order, we need to map back using sort indices
    # Create a scatter mask that marks positions of unique values in original order
    unsort_idx = sort_idx.argsort(dim=1)
    original_unique_mask = unique_mask.gather(1, unsort_idx)
    
    # Now we have a mask in original order - create padded tensor with all unique indices
    # Use cumsum to create position indices for scatter
    num_unique_per_row = original_unique_mask.sum(dim=1, keepdim=True)
    max_unique = min(max_actions, num_unique_per_row.max().item())
    
    # Create output tensor and position indices
    action_indices_padded = torch.zeros(batch_size_seq, max_unique, dtype=torch.long, device=combined_indices.device)
    action_masks = torch.zeros(batch_size_seq, max_unique, dtype=torch.bool, device=combined_indices.device)
    
    # FULLY VECTORIZED scatter using advanced indexing
    # Compute positions for scatter
    position_indices = original_unique_mask.cumsum(dim=1) - 1
    
    # Only scatter where we have unique values
    valid_positions = original_unique_mask & (position_indices < max_unique)
    
    # Get row and column indices for valid positions
    row_indices, col_indices = valid_positions.nonzero(as_tuple=True)
    positions = position_indices[row_indices, col_indices]
    
    # Scatter unique values to their positions in the output tensor
    action_indices_padded[row_indices, positions] = combined_indices[row_indices, col_indices]
    action_masks[row_indices, positions] = True
    
    # --- Step 2: Construct the Ideal Reward Distribution (PoE Model) ---
    # Get mentor (critic) probabilities
    mentor_probs = torch.nn.functional.softmax(critic_logits_flat, dim=-1)
    
    # FULLY VECTORIZED: Create reality expert distribution with ACTIVE TOKEN label smoothing
    # All operations stay on GPU - no loops, no CPU synchronization
    reality_probs = torch.zeros_like(mentor_probs)
    
    # Set ground truth probabilities for all positions in one operation
    batch_indices = torch.arange(batch_size_seq, device=reality_probs.device)
    reality_probs[batch_indices, ground_truth_flat] = 1.0 - label_smoothing_epsilon
    
    # VECTORIZED label smoothing distribution across active tokens
    # Count number of active tokens per sequence (for computing smoothing mass)
    num_active_per_seq = action_masks.sum(dim=1, keepdim=True).float()
    
    # Compute smoothing per token (epsilon divided by number of non-GT active tokens)
    # Handle edge case where only GT is active (num_active = 1)
    smoothing_per_token = torch.where(
        num_active_per_seq > 1,
        label_smoothing_epsilon / (num_active_per_seq - 1),
        torch.zeros_like(num_active_per_seq)
    )
    
    # Scatter smoothing mass to all active tokens using advanced indexing
    # Get indices of all active tokens
    active_rows, active_cols = action_masks.nonzero(as_tuple=True)
    active_token_ids = action_indices_padded[active_rows, active_cols]
    
    # Scatter add the smoothing values
    reality_probs[active_rows, active_token_ids] += smoothing_per_token[active_rows, 0]
    
    # Restore ground truth probability (overwrites any smoothing on GT token)
    reality_probs[batch_indices, ground_truth_flat] = torch.where(
        num_active_per_seq.squeeze() > 0,
        1.0 - label_smoothing_epsilon,
        1.0  # If no active tokens, GT gets full probability
    )
    
    # Combine experts using weighted geometric mean
    # P_ideal ∝ P_reality^0.7 * P_mentor^0.3
    ideal_probs = torch.pow(reality_probs, reality_weight) * torch.pow(mentor_probs, mentor_weight)
    
    # --- Step 3: Generate Positive Rewards ---
    # Extract raw probabilities for action space tokens only
    # Using our already-computed action_indices_padded and action_masks from above
    action_probs_raw = ideal_probs.gather(1, action_indices_padded)
    
    # Normalize ONLY over action space (not entire vocabulary!)
    # This concentrates 100% probability mass on our ~32 tokens
    masked_action_probs = action_probs_raw * action_masks.float()
    action_probs_sum = masked_action_probs.sum(dim=1, keepdim=True)
    action_probs_normalized = masked_action_probs / (action_probs_sum + 1e-8)
    
    # Implement mean thresholding: only reward tokens above mean
    # Calculate mean probability across valid actions
    valid_action_counts = action_masks.sum(dim=1, keepdim=True).float()
    mean_prob = action_probs_sum / (valid_action_counts + 1e-8)
    
    # Only reward tokens that are above mean (creates sparse rewards)
    above_mean_mask = (action_probs_normalized > mean_prob) & action_masks
    action_rewards = torch.where(
        above_mean_mask,
        action_probs_normalized * reward_scale,  # Above mean: get scaled reward
        torch.zeros_like(action_probs_normalized)  # Below mean: get zero
    )
    
    # Apply mask to ensure padded positions stay zero
    action_rewards = action_rewards * action_masks.float()
    
    # CRITICAL: Clamp rewards to max value and rescale others proportionally
    # This prevents gradient explosion while maintaining relative reward differences
    if max_reward_clamp is None:
        max_reward_clamp = globals().get('max_reward_clamp', 1.5)  # Use config value or default
    max_reward_per_seq = action_rewards.max(dim=1, keepdim=True)[0]
    
    # Only rescale if any reward exceeds the clamp threshold
    needs_rescaling = max_reward_per_seq > max_reward_clamp
    rescale_factor = torch.where(
        needs_rescaling,
        max_reward_clamp / (max_reward_per_seq + 1e-8),  # Proportional scaling factor
        torch.ones_like(max_reward_per_seq)  # No scaling needed
    )
    
    # Apply proportional rescaling to maintain relative differences
    action_rewards = action_rewards * rescale_factor
    
    # --- Step 3.5: Convert rewards to advantages (baseline: mean over valid actions)
    valid_action_counts = action_masks.sum(dim=1, keepdim=True).clamp(min=1)
    baseline = action_rewards.sum(dim=1, keepdim=True) / valid_action_counts
    action_advantages = (action_rewards - baseline) * action_masks.float()
    
    # --- Step 4: Calculate Policy Gradient Loss ---
    # Apply temperature scaling for exploration (replaces entropy regularization)
    # Higher temperature = more exploration, lower temperature = more exploitation
    temperature = 1.0 + entropy_coefficient  # Use entropy_coefficient to control exploration
    
    # full-vocab log-probs (B*T, V). this reintroduces the partition term so *all* logits get gradient.
    student_log_probs_full = torch.nn.functional.log_softmax(
        student_logits_flat / temperature, dim=-1
    )

    # gather only the action tokens' log-probs for the policy gradient (B*T, max_actions)
    student_log_probs_for_actions = student_log_probs_full.gather(1, action_indices_padded)
    
    # mask out padding in the action set
    student_log_probs_for_actions = student_log_probs_for_actions * action_masks.float()
    
    # Policy gradient loss: -sum(log_policy * advantage)
    # Detach advantages as they are fixed targets
    policy_gradient_loss = -(student_log_probs_for_actions * action_advantages.detach()).sum(dim=1)
    
    # Normalize by number of valid actions
    num_valid_actions = action_masks.sum(dim=1).float()
    policy_gradient_loss = policy_gradient_loss / (num_valid_actions + 1e-8)
    
    # --- Step 5: Combine Losses ---
    # Total loss = policy gradient loss (temperature scaling handles exploration)
    total_loss = policy_gradient_loss.mean()
    
    # --- Compute Additional Metrics for Logging ---
    # Calculate average rewards (only for valid actions)
    valid_rewards = action_rewards[action_masks]
    avg_reward = valid_rewards.mean().item() if valid_rewards.numel() > 0 else 0.0
    max_reward = valid_rewards.max().item() if valid_rewards.numel() > 0 else 0.0
    min_reward = valid_rewards.min().item() if valid_rewards.numel() > 0 else 0.0
    
    # Calculate average action space size
    avg_action_space_size = num_valid_actions.mean().item()
    
    # Return loss and metrics
    return total_loss, {
        'avg_reward': avg_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'avg_action_space_size': avg_action_space_size,
        'temperature': temperature  # Report temperature instead of entropy
    }

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    print("Initializing DDP...")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0, (
        f"{gradient_accumulation_steps=} must be divisible by {ddp_world_size=}"
    )
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Define data directory for dataset
data_dir = os.path.join("data", dataset)

# Dataset size tracking for epoch calculation
train_data_size = None
val_data_size = None
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
current_epoch = 0  # Initialize epoch counter
resume_run = init_from == "resume"
resume_wall_time = 0.0
wandb_run_id = None

tokenizer_name = _get_tokenizer_name()
vocab_size = _get_vocab_size_for_tokenizer(tokenizer_name)
bin_dtype = _get_bin_dtype_for_vocab_size(vocab_size)
print(f"Tokenizer: {tokenizer_name} | vocab_size={vocab_size} | bin_dtype={bin_dtype}")


_memmap_cache: dict[str, np.memmap] = {}


def _get_memmap(split: str) -> np.memmap:
    """Return (and cache) the memmap for a dataset split."""
    if split not in _memmap_cache:
        filename = os.path.join("data", dataset, f"{split}.bin")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset split '{split}' missing at {filename}")
        _memmap_cache[split] = np.memmap(filename, dtype=bin_dtype, mode="r")
    return _memmap_cache[split]


def get_dataset_size(split):
    """Get the size of a dataset split in tokens"""
    global train_data_size, val_data_size
    if split == "train" and train_data_size is not None:
        return train_data_size
    elif split == "val" and val_data_size is not None:
        return val_data_size
    
    data = _get_memmap(split)
    size = len(data)
    
    if split == "train":
        train_data_size = size
    else:
        val_data_size = size
    
    return size

# Calculate epoch information if dataset exists
try:
    if os.path.exists(os.path.join(data_dir, "train.bin")):
        train_tokens = get_dataset_size("train")
        # Number of sequences we can sample from the dataset
        num_sequences = train_tokens - block_size + 1
        iterations_per_epoch = math.ceil(train_tokens / tokens_per_iter)   # or use // for floor
        tokens_per_epoch = iterations_per_epoch * tokens_per_iter
                
        print(f"dataset has {train_tokens:,} tokens")
        print(f"iterations per epoch: {iterations_per_epoch:,}")
        print(f"tokens per epoch: {tokens_per_epoch:,}")
        
        # Handle max_epochs vs max_iters configuration
        if 'max_epochs' in globals() and max_epochs is not None:
            # Calculate max_iters from max_epochs
            max_iters = int(max_epochs * iterations_per_epoch)
            print(f"training for {max_epochs} epochs = {max_iters:,} iterations")
        else:
            # Show how many epochs the current max_iters represents
            print(f"with max_iters={max_iters}, training for {max_iters / iterations_per_epoch:.2f} epochs")
    else:
        iterations_per_epoch = None
        print(f"dataset not found at {data_dir}, cannot calculate epoch information")
        if 'max_epochs' in globals() and max_epochs is not None:
            print(f"WARNING: max_epochs specified but cannot calculate iterations without dataset")
except Exception as e:
    iterations_per_epoch = None
    print(f"could not calculate epoch information: {e}")

# Safety check: ensure max_iters has a value
if max_iters is None:
    if 'max_epochs' in globals() and max_epochs is not None:
        # If we couldn't calculate iterations_per_epoch, use a reasonable default
        print(f"WARNING: Cannot calculate iterations from epochs without dataset. Using default max_iters=10000")
        max_iters = 10000
    else:
        # Both max_iters and max_epochs are None - use default
        print(f"WARNING: Neither max_iters nor max_epochs specified. Using default max_iters=10000")
        max_iters = 10000


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
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

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched implementation
    a, b, c = (3.4445, -4.7750, 2.0315)  # Quintic coefficients for fast convergence
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    https://kellerjordan.github.io/posts/muon/
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    
    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    
    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            
            def update_prev():  # optimized Muon implementation
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"]
                        * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            
            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(
                        g, steps=group["ns_steps"]
                    ).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


# model init
if tokenizer_name == "qwen3":
    if init_from == "scratch":
        print("Initializing Qwen3 student model from scratch")
        # Use the canonical Qwen3-0.6B config as the architecture source-of-truth.
        # Only override runtime knobs that must match our training setup.
        qwen_cfg = dict(QWEN_CONFIG_06_B)
        qwen_cfg["context_length"] = block_size
        qwen_cfg["dtype"] = ptdtype
        # vocab_size should already match QWEN_CONFIG_06_B; keep it explicit to avoid drift
        qwen_cfg["vocab_size"] = vocab_size
        model_args = dict(model_type="qwen3", **qwen_cfg)
        model = Qwen3Model(qwen_cfg)
    elif init_from == "resume":
        print(f"Resuming Qwen3 training from {out_dir}")
        checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
        ckpt_path = os.path.join(out_dir, checkpoint_filename)

        if not os.path.exists(ckpt_path) and experiment_name:
            default_ckpt_path = os.path.join(out_dir, "ckpt.pt")
            if os.path.exists(default_ckpt_path):
                print(f"Experiment checkpoint '{checkpoint_filename}' not found, using default 'ckpt.pt'")
                ckpt_path = default_ckpt_path

        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        model_args = checkpoint_model_args.copy()
        model = Qwen3Model(model_args)
        
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        resume_wall_time = checkpoint.get("global_wall_time", 0.0)
        if "current_epoch" in checkpoint:
            current_epoch = checkpoint["current_epoch"]
        else:
            current_epoch = 0
    else:
        raise ValueError(f"Qwen3 student only supports init_from='scratch' or 'resume', got '{init_from}'")
else:
    print("Initializing GPT student model")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout,
    )
    if init_from == "scratch":
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
        ckpt_path = os.path.join(out_dir, checkpoint_filename)

        if not os.path.exists(ckpt_path) and experiment_name:
            default_ckpt_path = os.path.join(out_dir, "ckpt.pt")
            if os.path.exists(default_ckpt_path):
                print(f"Experiment checkpoint '{checkpoint_filename}' not found, using default 'ckpt.pt'")
                ckpt_path = default_ckpt_path

        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        resume_wall_time = checkpoint.get("global_wall_time", 0.0)
        if "current_epoch" in checkpoint:
            current_epoch = checkpoint["current_epoch"]
        else:
            current_epoch = 0
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size

model.to(device)

process_start_wall_time = time.time()


def current_global_wall_time() -> float:
    """Return absolute wall-clock seconds since original run start (persisted via checkpoints)."""
    return resume_wall_time + (time.time() - process_start_wall_time)

# Track resume metadata for logging/eval alignment
iter_display_offset = 1 if resume_run else 0
skip_initial_eval = resume_run

# Load critic model for AvataRL (supports HF via CRITIC_HF_MODEL or hf: prefix)
critic_source = os.environ.get("CRITIC_HF_MODEL") or critic_model_path
critic_model = load_critic_model(critic_source)
print(f"Teacher model loaded from {critic_source}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

# optimizer
if use_dual_optimizer:
    if tokenizer_name == "qwen3":
        raise ValueError("use_dual_optimizer is currently GPT-specific; disable it for Qwen3 student runs.")
    # Dual optimizer setup (Muon + Adam)
    # Collect parameters and group them
    hidden_matrix_params = [
        p for n, p in model.named_parameters() 
        if p.ndim >= 2 and "embed" not in n and n != "lm_head.weight"
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = []
    for n, p in model.named_parameters():
        if n == "lm_head.weight":
            head_params.append(p)
    
    # Initialize Adam optimizer for embeddings/head/scalars
    adam_params = [
        dict(params=head_params, lr=learning_rate * adam_head_lr_mult),  # Head: 36x higher LR
        dict(params=embed_params, lr=learning_rate * adam_embed_lr_mult),  # Embeddings: 100x higher LR
        dict(params=scalar_params, lr=adam_scalar_lr),  # Scalars: fixed LR
    ]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    
    # Initialize Muon optimizer for hidden matrices
    rank = ddp_rank if ddp else 0
    world_size = ddp_world_size
    optimizer2 = Muon(
        hidden_matrix_params, lr=muon_lr, momentum=muon_momentum, 
        ns_steps=muon_ns_steps, rank=rank, world_size=world_size
    )
    
    # Create optimizer list
    optimizers = [optimizer1, optimizer2]
    
    # Store initial learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    if init_from == "resume":
        # Load optimizer states for dual optimizers
        if isinstance(checkpoint["optimizer"], list):
            for opt, opt_state in zip(optimizers, checkpoint["optimizer"]):
                opt.load_state_dict(opt_state)
        else:
            # Backward compatibility: old checkpoint with single optimizer
            print("Warning: Loading single optimizer checkpoint into dual optimizer setup")
            # Only load into the first optimizer as a fallback
            optimizer1.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from iteration {iter_num}, best_val_loss={best_val_loss:.4f}")
else:
    # Original single optimizer setup
    if hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type
        )
    else:
        optimizer = _configure_optimizers_generic(
            model, weight_decay, learning_rate, (beta1, beta2), device_type
        )
    optimizers = [optimizer]  # Wrap in list for consistency
    
    # Store initial learning rate
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed from iteration {iter_num}, best_val_loss={best_val_loss:.4f}")

checkpoint = None  # free up memory

# wrap model into DDP container
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        bucket_cap_mb=ddp_bucket_cap_mb,
    )
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    t0_compile = time.time()
    model = torch.compile(model)  # requires PyTorch 2.0
    compile_time = time.time() - t0_compile
    print(f"Compilation completed in {compile_time:.2f} seconds")

# -----------------------------------------------------------------------------
# poor man's data loader

def get_batch(split):
    # Reuse cached memmap to avoid reopening the dataset every batch.
    data = _get_memmap(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    critic_model.eval()  # Ensure critic is in eval mode too
    
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        ce_losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                student_logits, _ = model(X, Y)
                
                with torch.no_grad():
                    critic_out = critic_model(X, Y) if hasattr(critic_model, "__call__") else critic_model(X)
                    critic_logits = _extract_logits(critic_out)
                
                loss, _ = compute_avatarl_loss(
                    student_logits, critic_logits, Y,
                    reality_weight=reality_weight,
                    mentor_weight=mentor_weight,
                    label_smoothing_epsilon=label_smoothing_epsilon,
                    reward_scale=reward_scale,
                    top_k=top_k,
                    entropy_coefficient=entropy_coefficient,
                    max_reward_clamp=max_reward_clamp
                )
                losses[k] = loss.item()

                ce_loss = torch.nn.functional.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    Y.view(-1)
                )
                ce_losses[k] = ce_loss.item()
        
        out[split] = losses.mean()
        out[f"{split}_ce"] = ce_losses.mean()
    
    model.train()
    critic_model.eval()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    resume_kwargs = {}
    if resume_run and wandb_run_id:
        resume_kwargs = {"id": wandb_run_id, "resume": "allow"}
    elif resume_run:
        print("Warning: resume requested but wandb_run_id missing in checkpoint; starting new W&B run.")

    wandb.init(project=wandb_project, name=wandb_run_name, config=config, **resume_kwargs)
    wandb_run_id = wandb.run.id if wandb.run else wandb_run_id

wait, warmup, active, repeat = 5, 5, 5, 2
num_steps = wait + warmup + active
if profile:
    print("Profiling NanoGPT model...")


def trace_handler(prof):
    print("Handling torch profiler trace...")
    task_id = os.environ["MODAL_TASK_ID"]
    rank = os.environ["RANK"]
    torch.profiler.tensorboard_trace_handler(f"/root/out/bench_log/{task_id}/{rank}")(
        prof
    )


profiler = (
    nullcontext()
    if not profile
    else torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    )
)

if speedrun and master_process:
    print("Speedrun mode enabled! 🏎️ 🏍️ 🐎 🏃‍♀️")

# progress bar (master only; show even if total is unknown)
# Note: iter_num is set during model initialization (0 for scratch, loaded value for resume)
progress_bar = None
if master_process:
    total_iters = max_iters if max_iters is not None else None
    progress_bar = tqdm(
        total=total_iters,
        initial=iter_num,  # Start from resumed iteration
        desc="train",
        leave=True,
        dynamic_ncols=True,
    )
    
# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
training_time_ms = 0
# Running averages for AvataRL metrics
running_avg_reward = 0.0
# Track another t0, separate from the original t0 which cares only about iter time.
# This t0 cares about overall training time, and is used in speedrun mode.
training_time_t0 = time.perf_counter()
with profiler:
    while True:
        display_iter = iter_num + iter_display_offset
        # Update epoch counter (use fractional epochs like train.py)
        if iterations_per_epoch is not None and iter_num > 0:
            current_epoch = iter_num / iterations_per_epoch
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        
        # Update learning rates for all optimizers
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = param_group["initial_lr"] * (lr / learning_rate)
        
        # Muon momentum warmup (only for dual optimizer mode)
        if use_dual_optimizer and iter_num < muon_warmup_iters:
            for group in optimizer2.param_groups:
                frac = min(iter_num / muon_warmup_iters, 1)
                group["momentum"] = (1 - frac) * muon_warmup_start_momentum + frac * muon_warmup_end_momentum

        # evaluate the loss on train/val sets and write checkpoints
        should_eval = (
            eval_interval
            and iter_num % eval_interval == 0
            and master_process
        )
        if should_eval and skip_initial_eval:
            skip_initial_eval = False
        elif should_eval:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - training_time_t0)

            losses = estimate_loss()
            epoch_str = f" (epoch {current_epoch:.2f})" if iterations_per_epoch else ""
            print(
                f"step {display_iter}{epoch_str}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val_ce_loss {losses['val_ce']:.4f} train_time:{training_time_ms:.0f}ms"
            )
            if speedrun and losses["val"] < speedrun_target_eval_loss:
                print(
                    f"Speedrun target eval loss {speedrun_target_eval_loss} reached! 🏆"
                )
                # we must teardown or else the program will hang waiting for other processes
                if ddp:
                    destroy_process_group()
                break

            if wandb_log:
                log_dict = {
                    "iter": display_iter,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "val/ce_loss": losses["val_ce"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "global_wall_time": current_global_wall_time(),
                }
                if iterations_per_epoch:
                    log_dict["epoch"] = current_epoch
                # Add AvataRL metrics if available
                if 'avatarl_metrics' in locals():
                    log_dict.update({
                        "avatarl/avg_reward": running_avg_reward,
                        "avatarl/instant_reward": avatarl_metrics['avg_reward'],
                    })
                wandb.log(log_dict, step=display_iter)
            if (
                losses["val"] < best_val_loss or always_save_checkpoint
            ) and not speedrun:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": [opt.state_dict() for opt in optimizers] if use_dual_optimizer else optimizers[0].state_dict(),
                        "model_args": model_args,
                        "model_type": tokenizer_name,  # Save model type (qwen3 or gpt2)
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "use_dual_optimizer": use_dual_optimizer,  # Save optimizer type for loading
                        "current_epoch": current_epoch,  # Save epoch information
                        "iterations_per_epoch": iterations_per_epoch,  # Save for consistency checks
                        "global_wall_time": current_global_wall_time(),
                        "wandb_run_id": wandb_run_id,
                    }
                    # Construct checkpoint filename with experiment name suffix
                    checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
                    checkpoint_path = os.path.join(out_dir, checkpoint_filename)
                    print(f"saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)
            # start the clock again
            torch.cuda.synchronize()
            training_time_t0 = time.perf_counter()
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # In DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code.
                # Looking at the source of that context manager, it just toggles this variable.
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                # Get student and critic logits - we need full sequence, so pass targets
                # but we'll ignore the returned loss and compute our own AvataRL loss
                student_logits, _ = model(X, Y)
                
                # Get critic logits for AvataRL
                with torch.no_grad():
                    critic_out = critic_model(X, Y) if hasattr(critic_model, "__call__") else critic_model(X)
                    critic_logits = _extract_logits(critic_out)
                
                # Compute AvataRL loss
                loss, avatarl_metrics = compute_avatarl_loss(
                    student_logits, critic_logits, Y,
                    reality_weight=reality_weight,
                    mentor_weight=mentor_weight,
                    label_smoothing_epsilon=label_smoothing_epsilon,
                    reward_scale=reward_scale,
                    top_k=top_k,
                    entropy_coefficient=entropy_coefficient,
                    max_reward_clamp=max_reward_clamp
                )
                
                # Calculate top-1 cross-entropy loss for logging
                with torch.no_grad():
                    top1_ce_loss = torch.nn.functional.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)), Y.view(-1)
                    )
                
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            # Unscale gradients for all optimizers
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # step all optimizers and scaler if training in fp16
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        
        # flush the gradients as soon as we can, no need for this memory anymore
        model.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps

            epoch_str = f" (epoch {current_epoch:.2f})" if iterations_per_epoch else ""
            out_str = f"iter {display_iter}{epoch_str}: loss {lossf:.4f}, ce_loss {top1_ce_loss.item():.4f}, time {dt * 1000:.2f}ms, wall_time {current_global_wall_time():.1f}s"
            
            # Update running averages for AvataRL metrics
            if 'avatarl_metrics' in locals():
                running_avg_reward = (1 - running_avg_alpha) * running_avg_reward + running_avg_alpha * avatarl_metrics['avg_reward']
                out_str += f", reward {running_avg_reward:.3f}"

            if local_iter_num >= 5:
                # In AvataRL, we do more computation than standard training:
                # 1. Additional critic forward pass (roughly +0.5x FLOPs since no backward)
                # 2. Computing gradients for top_k actions instead of 1 per position
                # 
                # FIXED: Correct FLOP accounting for AvataRL
                # Standard training: 1x forward + 2x backward = 3x FLOPs
                # AvataRL: 1x student forward + 2x student backward + 1x critic forward = 4x FLOPs
                # The action space operations (gather/scatter) are memory-bound, not compute-bound
                critic_overhead = 4.0 / 3.0  # 1.33x for critic forward (4x total / 3x standard)
                avatarl_overhead = 1.5  # Conservative estimate for AvataRL-specific ops (softmax, gather, rewards)
                
                # Combined multiplier: ~2x instead of incorrect 12x
                avatarl_multiplier = critic_overhead * avatarl_overhead  # ~2.0x
                # This reflects actual compute: critic forward + loss computation overhead
                # NOT top_k forward passes (which don't happen)
                
                if hasattr(raw_model, "estimate_mfu"):
                    mfu = raw_model.estimate_mfu(
                        batch_size * gradient_accumulation_steps * avatarl_multiplier, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                    out_str += f", mfu {running_mfu * 100:.2f}%"

            print(out_str)

        iter_num += 1
        local_iter_num += 1
        if progress_bar is not None:
            progress_bar.update(1)

        if profile:
            profiler.step()

        # termination conditions
        if max_iters is not None and iter_num > max_iters:
            break

if ddp:
    destroy_process_group()
if progress_bar is not None:
    progress_bar.close()