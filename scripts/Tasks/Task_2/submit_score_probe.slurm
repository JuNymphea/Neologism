#!/bin/bash -l
#SBATCH --job-name=score_probe
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Score every control word in every Task 2 slot, then calibrate.
#
#   sbatch scripts/Tasks/Task_2/submit_score_probe.slurm
#   sbatch --export=ALL,BATCH=128,MIN_AUC=0.65 scripts/Tasks/Task_2/submit_score_probe.slurm
#
# 30 slots x 900 words = 27,000 short forward passes -- teacher forcing only, no
# generation, so this is far cheaper than eval and l40s is ample for a 4B model.
# Keep the ALL: --export with a bare list replaces the default environment
# instead of adding to it.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

TASK=scripts/Tasks/Task_2
BATCH=${BATCH:-64}
MIN_AUC=${MIN_AUC:-0.60}
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}

echo "=== $(date) | node $(hostname) | model $MODEL | batch $BATCH ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The expensive part runs once and is cached in out/surprisal.json, so the
# calibration below -- and any later change of threshold -- needs no GPU.
python "$TASK/score_probe.py" \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    --slots "$TASK/out/slots_flat.json" \
    --calibration "$TASK/out/control_calibration.json" \
    --validation "$TASK/out/control_validation.json" \
    --out "$TASK/out/surprisal.json"

python "$TASK/calibrate_probe.py" \
    --surprisal "$TASK/out/surprisal.json" \
    --out "$TASK/out/calibration.json" \
    --min-auc "$MIN_AUC"

crc-job-stats
