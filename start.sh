#!/bin/bash

# start.sh - Run nanoGPT training locally with torchrun
# Supports both regular training (train.py) and AvataRL training (avatarl.py)
# Based on Modal configuration from modal_train.py

set -e  # Exit on error

echo "=== AvataRL Local Training Script ==="
echo

# Parse command line arguments
GPUS=${1:-1}  # Default to 1 GPU if not specified
MODE=${2:-"single"}  # Default to single node
SCRIPT=${3:-"avatarl"}  # Default to avatarl, can be "train" for regular training
# Optional controls via env:
#   TOKENIZER=gpt2|qwen3 (default gpt2) for data prep
#   CRITIC_HF_MODEL=Qwen/Qwen3-0.6B-Base to use HF critic
#   USE_4BIT_CRITIC=true/false to override config
TOKENIZER=${TOKENIZER:-"gpt2"}
# Export so python training script can see it
export TOKENIZER

echo "Configuration:"
echo "- GPUs per node: $GPUS"
echo "- Mode: $MODE"
echo "- Script: $SCRIPT"
echo "- Tokenizer: $TOKENIZER"
echo "- HF Critic (optional): ${CRITIC_HF_MODEL:-none}"
echo

# Determine which training script to use
if [ "$SCRIPT" = "train" ]; then
    TRAIN_SCRIPT="train.py"
else
    TRAIN_SCRIPT="avatarl.py"
fi

# Install dependencies if 
# use uv to install dependencies
echo "=== Checking Dependencies ==="
# check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -sSL https://astral.sh/uv/install.sh | sh
fi
uv init

uv add torch==2.6.0 \
    transformers==4.51.3 \
    datasets==3.6.0 \
    tiktoken==0.9.0 \
    tokenizers==0.21.0 \
    wandb==0.19.11 \
    tqdm==4.67.1 \
    python-dotenv==1.2.1 \

uv sync

# Check for data (tokenizer-aware)
DATA_DIR="data/openwebtext"
if [ "$TOKENIZER" = "gpt2" ]; then
    if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
        echo
        echo "Data not found at $DATA_DIR, downloading GPT-2 tokenized bins from HuggingFace..."
        
        mkdir -p "$DATA_DIR"
        
        if command -v wget &> /dev/null; then
            echo "Downloading train.bin (604 kB)..."
            wget -q --show-progress -O "$DATA_DIR/train.bin" \
                "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/train.bin" || {
                echo "ERROR: Failed to download train.bin"
                exit 1
            }
            
            echo "Downloading val.bin (72.1 kB)..."
            wget -q --show-progress -O "$DATA_DIR/val.bin" \
                "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/val.bin" || {
                echo "ERROR: Failed to download val.bin"
                exit 1
            }
        elif command -v curl &> /dev/null; then
            echo "Downloading train.bin (604 kB)..."
            curl -L -o "$DATA_DIR/train.bin" \
                "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/train.bin" || {
                echo "ERROR: Failed to download train.bin"
                exit 1
            }
            
            echo "Downloading val.bin (72.1 kB)..."
            curl -L -o "$DATA_DIR/val.bin" \
                "https://huggingface.co/datasets/TokenBender/avataRL/resolve/main/val.bin" || {
                echo "ERROR: Failed to download val.bin"
                exit 1
            }
        else
            echo "ERROR: Neither wget nor curl is installed. Please install one to download data."
            exit 1
        fi
        
        echo "Data downloaded successfully!"
    fi
else
    echo
    echo "Preparing OpenWebText with tokenizer: $TOKENIZER (this may take a while)..."
    python3 prepare_data.py --dataset openwebtext --tokenizer "$TOKENIZER"
fi

# Check for critic model (only needed for AvataRL)
if [ "$SCRIPT" = "avatarl" ]; then
    echo
    echo "=== Checking Critic Model ==="

    if [ -n "${CRITIC_HF_MODEL:-}" ]; then
        echo "Using HuggingFace critic model: $CRITIC_HF_MODEL"
        echo "No local checkpoint download needed."
        export CRITIC_HF_MODEL
    else
        # Extract critic model path from config
        CRITIC_PATH=$(python3 -c "
import sys
sys.path.insert(0, 'config')
from train_avatarl import critic_model_path
print(critic_model_path)
" 2>&1) || {
            echo "ERROR: Failed to read critic_model_path from config/train_avatarl.py"
            echo "Error details: $CRITIC_PATH"
            exit 1
        }
        
        echo "Config expects critic at: $CRITIC_PATH"
        
        CRITIC_FILE=$(basename "$CRITIC_PATH")
        echo "Critic filename: $CRITIC_FILE"
        
        if [ ! -f "$CRITIC_PATH" ]; then
            echo "Critic model not found at $CRITIC_PATH"
            echo "Downloading $CRITIC_FILE from HuggingFace..."
            
            mkdir -p $(dirname "$CRITIC_PATH")
            
            if [ -n "$HF_TOKEN" ]; then
                AUTH_HEADER="Authorization: Bearer $HF_TOKEN"
            else
                AUTH_HEADER=""
            fi
            
            if command -v wget &> /dev/null; then
                if [ -n "$AUTH_HEADER" ]; then
                    wget -q --show-progress --header="$AUTH_HEADER" -O "$CRITIC_PATH" \
                        "https://huggingface.co/TokenBender/avataRL-critic/resolve/main/$CRITIC_FILE" || {
                        echo "ERROR: Failed to download critic model $CRITIC_FILE"
                        echo "Please check your internet connection or download manually from:"
                        echo "https://huggingface.co/TokenBender/avataRL-critic"
                        echo "If the repo is private, set HF_TOKEN environment variable"
                        exit 1
                    }
                else
                    wget -q --show-progress -O "$CRITIC_PATH" \
                        "https://huggingface.co/TokenBender/avataRL-critic/resolve/main/$CRITIC_FILE" || {
                        echo "ERROR: Failed to download critic model $CRITIC_FILE"
                        echo "Please check your internet connection or download manually from:"
                        echo "https://huggingface.co/TokenBender/avataRL-critic"
                        echo "If the repo is private, set HF_TOKEN environment variable"
                        exit 1
                    }
                fi
            elif command -v curl &> /dev/null; then
                if [ -n "$AUTH_HEADER" ]; then
                    curl -L --progress-bar -H "$AUTH_HEADER" -o "$CRITIC_PATH" \
                        "https://huggingface.co/TokenBender/avataRL-critic/resolve/main/$CRITIC_FILE" || {
                        echo "ERROR: Failed to download critic model $CRITIC_FILE"
                        echo "Please check your internet connection or download manually from:"
                        echo "https://huggingface.co/TokenBender/avataRL-critic"
                        echo "If the repo is private, set HF_TOKEN environment variable"
                        exit 1
                    }
                else
                    curl -L --progress-bar -o "$CRITIC_PATH" \
                        "https://huggingface.co/TokenBender/avataRL-critic/resolve/main/$CRITIC_FILE" || {
                        echo "ERROR: Failed to download critic model $CRITIC_FILE"
                        echo "Please check your internet connection or download manually from:"
                        echo "https://huggingface.co/TokenBender/avataRL-critic"
                        echo "If the repo is private, set HF_TOKEN environment variable"
                        exit 1
                    }
                fi
            else
                echo "ERROR: Neither wget nor curl is installed. Please install one to download the critic model."
                exit 1
            fi
            
            echo "Critic model downloaded successfully!"
        else
            echo "Critic model found at $CRITIC_PATH"
        fi
    fi
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Optional: Disable Infiniband if not available
if [ "$DISABLE_IB" = "1" ]; then
    export NCCL_IB_DISABLE=1
    echo "Infiniband disabled"
fi

# Run training
echo
if [ "$SCRIPT" = "train" ]; then
    echo "=== Starting Regular Training ==="
else
    echo "=== Starting AvataRL Training ==="
fi
echo

case $MODE in
    "single")
        if [ $GPUS -eq 1 ]; then
            # Single GPU - run directly with python
            echo "Running on single GPU..."
            python3 $TRAIN_SCRIPT --compile=False
        else
            # Multi-GPU single node
            echo "Running on $GPUS GPUs (single node)..."
            torchrun --standalone --nproc_per_node=$GPUS $TRAIN_SCRIPT
        fi
        ;;
    
    "master")
        # Multi-node master
        if [ -z "$MASTER_ADDR" ]; then
            MASTER_ADDR=$(hostname -I | awk '{print $1}')
        fi
        MASTER_PORT=${MASTER_PORT:-1234}
        NNODES=${NNODES:-2}
        
        echo "Starting master node..."
        echo "Master address: $MASTER_ADDR:$MASTER_PORT"
        echo "Total nodes: $NNODES"
        
        torchrun \
            --nproc_per_node=$GPUS \
            --nnodes=$NNODES \
            --node_rank=0 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT
        ;;
    
    "worker")
        # Multi-node worker
        if [ -z "$MASTER_ADDR" ]; then
            echo "Error: MASTER_ADDR must be set for worker nodes"
            exit 1
        fi
        MASTER_PORT=${MASTER_PORT:-1234}
        NNODES=${NNODES:-2}
        NODE_RANK=${NODE_RANK:-1}
        
        echo "Starting worker node $NODE_RANK..."
        echo "Connecting to master: $MASTER_ADDR:$MASTER_PORT"
        
        torchrun \
            --nproc_per_node=$GPUS \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./start.sh [num_gpus] [mode] [script]"
        echo "  num_gpus: Number of GPUs to use (default: 1)"
        echo "  mode: single, master, or worker (default: single)"
        echo "  script: avatarl or train (default: avatarl)"
        echo
        echo "Examples:"
        echo "  ./start.sh                      # Single GPU with avatarl.py"
        echo "  ./start.sh 1 single train       # Single GPU with train.py"
        echo "  ./start.sh 4                    # 4 GPUs with avatarl.py"
        echo "  ./start.sh 4 single train       # 4 GPUs with train.py"
        echo "  ./start.sh 8 master             # Master node with 8 GPUs, avatarl.py"
        echo "  ./start.sh 8 master train       # Master node with 8 GPUs, train.py"
        echo "  MASTER_ADDR=10.0.0.1 ./start.sh 8 worker  # Worker node"
        exit 1
        ;;
esac

echo
echo "=== Training Started ==="
echo "Check wandb for training progress: https://wandb.ai"
echo "Output directory: out-avatarl/"
echo "Checkpoints will be saved as: out-avatarl/ckpt_avatarl_pretrain.pt"