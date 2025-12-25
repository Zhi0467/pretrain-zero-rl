#!/bin/bash
#SBATCH --job-name=pretrain_zero_rl
#SBATCH --output=logs/%x_%j.log     # %x=job-name, %j=job-id
#SBATCH --error=logs/%x_%j.err      # Separate error log is often helpful
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-user=zhw119@ucsd.edu
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR" || exit

# run the training script
bash start.sh "$@"