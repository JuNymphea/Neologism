#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-29%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --requeue

# a_1..a_10, n_1..n_10, v_1..v_10 against all five templates: 150 runs.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch --export=ALL,MODEL_KEY=gemma scripts/train/submit_train_sweep.slurm
#   sbatch --export=ALL,MODEL_KEY=qwen  scripts/train/submit_train_sweep.slurm
#
# MODEL_KEY is required and picks train_<key>.py, which holds that model's settings
# (batch size, hinge threshold, chat template, model path, output directories).
# Anything exported below overrides those settings; leave it unset to use them.
#
# One array task per concept, five runs each, about 45 minutes. Array tasks are
# scheduled independently -- three GPUs are never needed at once -- and the %3 caps
# how many run together. Short tasks also backfill into gaps a seven-hour job could
# never fit; that, not the number of GPUs, is what decides the wait.
#
#   more at once:        sbatch --array=0-29%6 ...
#   one part of speech:  sbatch --array=0-9 ...    (0-9 a, 10-19 n, 20-29 v)
#   other concepts:      sbatch --export=ALL,FIRST=11 --array=0-29 ...
#
# The 9 GB of weights load once per task rather than once per run. Runs that
# already have an embedding_final.pt are skipped, so a requeued or resubmitted
# task resumes instead of repeating work.
#
#   different outputs:   sbatch --export=ALL,MODEL_KEY=...,CKPT_DIR=...,RES_DIR=... ...
#   different flags:     sbatch --export=ALL,MODEL_KEY=...,BATCH=8,TRAIN_ARGS="--lambda_h 0" ...
#   another checkpoint:  sbatch --export=ALL,MODEL_KEY=qwen,MODEL=/path/to/Qwen3-8B ...
#   Chinese data:        sbatch --export=ALL,MODEL_KEY=qwen,DATA_LANG=zh ...
#     (DATA_LANG, not LANG: LANG is the shell's locale)

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

case "${MODEL_KEY:-}" in
    gemma|qwen) ;;
    *) echo "ERROR: export MODEL_KEY=gemma or MODEL_KEY=qwen (got '${MODEL_KEY:-}')"; exit 1 ;;
esac
ENTRY=scripts/train/train_${MODEL_KEY}.py

POSLIST=(a n v)
PER_POS=${PER_POS:-10}
FIRST=${FIRST:-1}
POS=${POSLIST[$((SLURM_ARRAY_TASK_ID / PER_POS))]}
CONCEPT="${POS}_$((FIRST + SLURM_ARRAY_TASK_ID % PER_POS))"

# only what was exported overrides the model's own settings in $ENTRY
EXTRA=()
[[ -n "${MODEL:-}" ]]    && EXTRA+=(--model_name "$MODEL")
[[ -n "${BATCH:-}" ]]    && EXTRA+=(--batch_size "$BATCH")
[[ -n "${CKPT_DIR:-}" ]] && EXTRA+=(--output_dir "$CKPT_DIR")
[[ -n "${RES_DIR:-}" ]]  && EXTRA+=(--results_dir "$RES_DIR")
[[ -n "${DATA_LANG:-}" ]] && EXTRA+=(--lang "$DATA_LANG")

echo "=== $(date) | node $(hostname) | $ENTRY | task $SLURM_ARRAY_TASK_ID -> $CONCEPT ==="
echo "    overrides: ${EXTRA[*]:-none} ${TRAIN_ARGS:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# the [config] line train prints first records every setting actually used
# shellcheck disable=SC2086  -- TRAIN_ARGS is meant to split into several flags
python "$ENTRY" \
    --concept "$CONCEPT" \
    --template unbiased verb noun adj mixed \
    ${EXTRA[@]+"${EXTRA[@]}"} ${TRAIN_ARGS:-}

crc-job-stats
