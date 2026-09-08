#!/bin/bash
#SBATCH --job-name=neologism
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL

# One array task per prompt template. Submit from the directory that contains
# neologism/ (the same place the old train_neologism.sh assumed), and create the
# log directory first:
#
#   mkdir -p logs
#   sbatch neologism/scripts/train/submit_train.slurm
#
# Peak usage is ~15 GB at batch 8, so one L40S (48 GB) is comfortable. Swap the
# partition for a100 (40 GB) if l40s is busy -- both fit.

set -euo pipefail

TEMPLATES=(unbiased verb noun adj mixed)
TEMPLATE=${TEMPLATES[$SLURM_ARRAY_TASK_ID]}

CONCEPT=a_25
MODEL=neologism/model/google/gemma-3-4b-it
TRAIN=neologism/scripts/train/train_neologism.py

# Slurm does not source ~/.bashrc, so conda is not on PATH by default
source ~/.bashrc
conda activate disrpt

cd "$SLURM_SUBMIT_DIR"

echo "=== $(date) | node $(hostname) | template $TEMPLATE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$TRAIN" \
    --concept "$CONCEPT" \
    --template "$TEMPLATE" \
    --model_name "$MODEL" \
    --batch_size 8

# CRCD's own accounting summary: prints the resources this job actually used,
# which is what you want to size the next submission from
crc-job-stats
