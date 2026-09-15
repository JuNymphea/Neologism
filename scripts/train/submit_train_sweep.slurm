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
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

# a_1..a_10, n_1..n_10, v_1..v_10 against all five templates: 150 runs.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch scripts/train/submit_train_sweep.slurm
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
# Defaults are the settled configuration: N(0, 0.02) init, hinge lambda_h 0.1,
# batch size 1, training prompt without the chat template. On a_1 adj this lifted
# the concept-minus-normal judge gap from 0.39 (batch 8) to 1.40. Outputs go to
# their own directories because the batch-8 vectors still sit in checkpoints/,
# and the skip-if-present rule would otherwise pass over every run.
#
#   different outputs:   sbatch --export=ALL,CKPT_DIR=...,RES_DIR=... ...
#   different flags:     sbatch --export=ALL,BATCH=8,TRAIN_ARGS="--lambda_h 0" ...

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

# Which model: the one setup_env.sh set up, unless MODEL is exported, e.g.
#   sbatch --export=ALL,MODEL=/ix/$(id -gn)/$USER/model/Qwen/Qwen3-4B ...
# TAG is the prefix every run, result and tokenizer of that model is filed under.
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
TAG=$(python scripts/train/naming.py tag "$MODEL")

POSLIST=(a n v)
PER_POS=${PER_POS:-10}
FIRST=${FIRST:-1}
BATCH=${BATCH:-1}
TRAIN_ARGS=${TRAIN_ARGS:---no_chat_template --lambda_h 0.1}
CKPT_DIR=${CKPT_DIR:-scripts/train/checkpoints_bs1hinge}
RES_DIR=${RES_DIR:-scripts/train/results_bs1hinge}

POS=${POSLIST[$((SLURM_ARRAY_TASK_ID / PER_POS))]}
CONCEPT="${POS}_$((FIRST + SLURM_ARRAY_TASK_ID % PER_POS))"

echo "=== $(date) | node $(hostname) | $TAG | task $SLURM_ARRAY_TASK_ID -> $CONCEPT ==="
echo "    batch $BATCH | TRAIN_ARGS '$TRAIN_ARGS' | $CKPT_DIR | $RES_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086  -- TRAIN_ARGS is meant to split into several flags
python scripts/train/train_neologism.py \
    --concept "$CONCEPT" \
    --template unbiased verb noun adj mixed \
    --model_name "$MODEL" \
    --batch_size "$BATCH" \
    --output_dir "$CKPT_DIR" \
    --results_dir "$RES_DIR" \
    $TRAIN_ARGS

crc-job-stats
