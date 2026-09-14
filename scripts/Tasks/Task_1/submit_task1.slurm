#!/bin/bash -l
#SBATCH --job-name=task1
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Task 1: ask the model outright which part of speech a word is, and read the
# answer off its probabilities.
#
#   sbatch scripts/Tasks/Task_1/submit_task1.slurm
#   sbatch --export=ALL,WORDS=/path/to/words.json scripts/Tasks/Task_1/submit_task1.slurm
#
# 234 words x 24 surface forms = ~5,600 short teacher-forced passes -- 24 being
# the union of every variant set the sensitivity report walks, of which the
# main set is 8. Far cheaper than Task 2, which scores the same words in 109
# candidate slots.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

TASK=scripts/Tasks/Task_1
BATCH=${BATCH:-64}
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
# The Task 2 final-test words by default, so the two probes are measured on the
# same items and their accuracies can be compared directly.
WORDS=${WORDS:-scripts/Tasks/Task_2/out/control_finaltest.json}

echo "=== $(date) | node $(hostname) | model $MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The sensitivity report is free: every variant set draws from the same union
# of surface forms, which is scored once, so reporting them is re-aggregation.
python "$TASK/run_task1.py" \
    --model "$MODEL" \
    --words "$WORDS" \
    --batch-size "$BATCH" \
    --report-sensitivity \
    --out "$TASK/out/task1.json"

# All 900 control words, not just final-test. Task 1 fits nothing, so the
# calibration and probe-dev splits were never spent on it and are just as
# clean -- three times the sample, and the item-level comparison against
# Task 2 on the words the two probes share.
python "$TASK/eval_on_task2_words.py" \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    --task2-out scripts/Tasks/Task_2/out \
    --out "$TASK/out/task1_on_task2_words.json"

crc-job-stats
