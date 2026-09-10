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

# Score the corpus candidate pool, screen it into a frozen 10/10/10 probe set,
# then report on the held-out control words.
#
#   sbatch scripts/Tasks/Task_2/submit_score_probe.slurm
#   sbatch --export=ALL,BATCH=128,MIN_AUC=0.65 scripts/Tasks/Task_2/submit_score_probe.slurm
#
# ~109 candidates x 900 words = ~98,000 short forward passes -- teacher forcing
# only, no generation, so this is still far cheaper than eval and l40s is ample
# for a 4B model.
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
    --slots "$TASK/out/candidates.json" \
    --calibration "$TASK/out/control_calibration.json" \
    --probedev "$TASK/out/control_probedev.json" \
    --finaltest "$TASK/out/control_finaltest.json" \
    --out "$TASK/out/surprisal.json"

# Walk the corpus ranking, keep candidates whose probe-dev AUC clears the bar,
# stop at ten per category, and freeze. The threshold is the same for all three.
python "$TASK/select_slots.py" \
    --surprisal "$TASK/out/surprisal.json" \
    --out "$TASK/out/probe_set.json" \
    --min-auc "$MIN_AUC"

# Measures the frozen set only -- it can no longer add or drop anything.
python "$TASK/calibrate_probe.py" \
    --surprisal "$TASK/out/surprisal.json" \
    --probe-set "$TASK/out/probe_set.json" \
    --out "$TASK/out/calibration.json"

crc-job-stats
