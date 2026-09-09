#!/bin/bash -l
#SBATCH --job-name=score_probe
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# 900 words x 30 slots = 27,000 short forward passes, batched 64 at a time. That is
# inference on very short sequences, so it runs on l40s and leaves the A100s for
# training.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch scripts/Tasks/Task_2/submit_score_probe.slurm
#
# Override with --export=ALL,BATCH=32 or MODEL=<path>.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || REPO="${SLURM_SUBMIT_DIR:-$PWD}/neologism"
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

# The script's default is the hub id google/gemma-3-4b-it. Compute nodes generally
# have no outbound network, and the weights are already on /ix, so point at those.
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
BATCH=${BATCH:-64}

echo "=== $(date) | node $(hostname) | batch $BATCH ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/Tasks/Task_2/score_probe.py \
    --model "$MODEL" \
    --batch-size "$BATCH"

crc-job-stats
