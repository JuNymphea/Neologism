#!/bin/bash -l
#SBATCH --job-name=eval
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Inference only, so it runs on l40s rather than competing for A100s: generation is
# memory-bandwidth bound and 48 GB is more than enough for a 4B model plus KV cache.
#
#   sbatch --export=RUN=gemma_a_25_unbiased scripts/train/submit_eval.slurm
#
# RUN names the checkpoint directory under scripts/train/checkpoints/.

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

RUN=${RUN:-gemma_a_25_unbiased}
CKPT=scripts/train/checkpoints/$RUN

echo "=== $(date) | node $(hostname) | run $RUN ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# new_token and template are read back out of the embedding file, so they cannot
# drift from what training used
python scripts/train/eval_neologism.py \
    --concept a_25 \
    --concept_model_path "${NEOLOGISM_MODEL:?source env.sh first}" \
    --concept_tokenizer_path "$CKPT/tokenizer" \
    --embedding_path "$CKPT/embedding/embedding_final.pt" \
    --max_samples 100 \
    --batch_size 8

crc-job-stats
