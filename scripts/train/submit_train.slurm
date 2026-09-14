#!/bin/bash -l
#SBATCH --job-name=neologism
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
# BEGIN/END/FAIL without ARRAY_TASKS means Slurm mails once for the array as a
# whole, not once per template. Add ARRAY_TASKS if you want all five separately
# (that is 10+ messages).
#SBATCH --mail-type=BEGIN,END,FAIL

# One array task per prompt template. The repo is located automatically, so it does
# not matter whether you submit from inside the clone or from its parent; the log
# directory is relative to wherever you do submit from:
#
#   cd ~/neologism && mkdir -p logs
#   sbatch scripts/train/submit_train.slurm
#
# Peak usage is ~13 GB at batch 8 (worst-case sequence length), so one A100-40GB
# has plenty of room. l40s (48 GB) also fits and sometimes has a shorter queue --
# check with: squeue -M gpu -p a100 | wc -l   vs   squeue -M gpu -p l40s | wc -l

set -euo pipefail

# Locate the repo instead of assuming where sbatch was run from: walk up from the
# submit directory, then try one level down, so submitting from inside the clone or
# from its parent both work.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || REPO="${SLURM_SUBMIT_DIR:-$PWD}/neologism"
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

TEMPLATES=(unbiased verb noun adj mixed)
TEMPLATE=${TEMPLATES[$SLURM_ARRAY_TASK_ID]}

CONCEPT=a_25
TRAIN=scripts/train/train_neologism.py

# written by setup_env.sh: loads the torch module, activates the venv on /ix, and
# points HF_HOME away from the 75 GB home quota. It has to come before MODEL is read,
# which is where NEOLOGISM_MODEL is defined.
source scripts/train/env.sh

# Which model: the one setup_env.sh set up, unless MODEL is exported, e.g.
#   sbatch --export=ALL,MODEL=/ix/$(id -gn)/$USER/model/Qwen/Qwen3-4B ...
# TAG is the prefix every run, result and tokenizer of that model is filed under.
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
TAG=$(python scripts/train/naming.py tag "$MODEL")

echo "=== $(date) | node $(hostname) | $TAG | template $TEMPLATE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$TRAIN" \
    --concept "$CONCEPT" \
    --template "$TEMPLATE" \
    --model_name "$MODEL" \
    --batch_size 8

# CRCD's own accounting summary: prints the resources this job actually used,
# which is what you want to size the next submission from
crc-job-stats
