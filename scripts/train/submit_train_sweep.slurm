#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# One GPU per part of speech: a_1..a_10, n_1..n_10, v_1..v_10, each against all five
# templates. 10 concepts x 5 templates = 50 runs per job, 150 in total.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch scripts/train/submit_train_sweep.slurm
#
# A single part of speech:            sbatch --array=0 ...   (0=a, 1=n, 2=v)
# A different range of concepts:      sbatch --export=ALL,FIRST=11,LAST=20 ...
#
# The 9 GB of weights are loaded once for all 50 runs rather than once per run --
# that alone was 8 of the 15 hours a naive loop would take. Runs that already have
# an embedding_final.pt are skipped, so resubmitting after a wall-clock limit
# resumes instead of starting over.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

POSLIST=(a n v)
POS=${POSLIST[$SLURM_ARRAY_TASK_ID]}
FIRST=${FIRST:-1}
LAST=${LAST:-10}
BATCH=${BATCH:-8}

CONCEPTS=()
for i in $(seq "$FIRST" "$LAST"); do CONCEPTS+=("${POS}_${i}"); done

echo "=== $(date) | node $(hostname) | pos $POS | ${CONCEPTS[*]} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train/train_neologism.py \
    --concept "${CONCEPTS[@]}" \
    --template unbiased verb noun adj mixed \
    --model_name "${NEOLOGISM_MODEL:?source env.sh first}" \
    --batch_size "$BATCH"

crc-job-stats
