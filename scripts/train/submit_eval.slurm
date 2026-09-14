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
#   sbatch --export=ALL,RUN=gemma_a_25_unbiased scripts/train/submit_eval.slurm
#   sbatch --export=ALL,MODEL=/ix/.../model/Qwen/Qwen3-4B,RUN=qwen_a_25_unbiased ...
#
# RUN names the checkpoint directory under scripts/train/checkpoints/. Keep the ALL:
# --export with a bare list replaces the default instead of adding to it, which would
# drop the rest of the submitting environment.

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

# Which model: the one setup_env.sh set up, unless MODEL is exported, e.g.
#   sbatch --export=ALL,MODEL=/ix/$(id -gn)/$USER/model/Qwen/Qwen3-4B ...
# TAG is the prefix every run, result and tokenizer of that model is filed under.
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
TAG=$(python scripts/train/naming.py tag "$MODEL")

RUN=${RUN:-${TAG}_a_25_unbiased}
CKPT=scripts/train/checkpoints/$RUN
# qwen_a_25_unbiased -> a_25: drop the model tag in front and the template behind
CONCEPT=${CONCEPT:-$(echo "$RUN" | sed -E 's/^[^_]+_//; s/_[^_]+$//')}
# Generation re-reads all 8 GB of weights for every decoding step regardless of batch
# size, so a larger batch amortises that and raises throughput almost linearly until
# the KV cache runs out of room. 46 GB leaves plenty of headroom past 8.
BATCH=${BATCH:-24}
MAXNEW=${MAXNEW:-4096}
SAMPLES=${SAMPLES:-100}

echo "=== $(date) | node $(hostname) | $TAG | run $RUN | batch $BATCH ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# new_token and template are read back out of the embedding file, so they cannot
# drift from what training used
python scripts/train/eval_neologism.py \
    --concept "$CONCEPT" \
    --concept_model_path "$MODEL" \
    --embedding_path "$CKPT/embedding/embedding_final.pt" \
    --max_samples "$SAMPLES" \
    --max_new_tokens "$MAXNEW" \
    --batch_size "$BATCH"

crc-job-stats
