#!/bin/bash -l
#SBATCH --job-name=probe
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL

# One concept x one template, trained with extra flags and evaluated, into its own
# directories -- for A/B probes of a training change without touching the sweep's
# outputs. Everything comes in through --export, so no quoting inside --wrap:
#
#   sbatch --export=ALL,RUNTAG=hinge_raw,CONCEPT=a_1,TEMPLATE=adj,\
#          TRAIN_ARGS="--lambda_h 0.1 --no_chat_template" scripts/train/submit_probe.slurm
#
#   RUNTAG      names checkpoints_<RUNTAG>/ and results_<RUNTAG>/          (required)
#   CONCEPT     default a_1        TEMPLATE  default adj
#   TRAIN_ARGS  extra train_neologism.py flags, e.g. "--lambda_h 0.1"    (default none)
#   MODEL       default $NEOLOGISM_MODEL from env.sh
#   BATCH / EVAL_BATCH / MAXNEW / SAMPLES   default 8 / 48 / 1536 / 100

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

: "${RUNTAG:?export RUNTAG=<name> -- it names checkpoints_<name>/ and results_<name>/}"
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
TAG=$(python scripts/train/naming.py tag "$MODEL")
CONCEPT=${CONCEPT:-a_1}
TEMPLATE=${TEMPLATE:-adj}
TRAIN_ARGS=${TRAIN_ARGS:-}
BATCH=${BATCH:-8}
EVAL_BATCH=${EVAL_BATCH:-48}
MAXNEW=${MAXNEW:-1536}
SAMPLES=${SAMPLES:-100}

CKPT=scripts/train/checkpoints_$RUNTAG
RES=scripts/train/results_$RUNTAG
RUN=${TAG}_${CONCEPT}_${TEMPLATE}

echo "=== $(date) | node $(hostname) | $RUN | RUNTAG=$RUNTAG | TRAIN_ARGS='$TRAIN_ARGS' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== train ==="
# shellcheck disable=SC2086  -- TRAIN_ARGS is meant to split into several flags
python scripts/train/train_neologism.py \
    --concept "$CONCEPT" --template "$TEMPLATE" --model_name "$MODEL" \
    --batch_size "$BATCH" --output_dir "$CKPT" --results_dir "$RES" $TRAIN_ARGS

echo "=== eval ==="
python scripts/train/eval_neologism.py \
    --concept "$CONCEPT" --concept_model_path "$MODEL" \
    --embedding_path "$CKPT/$RUN/embedding/embedding_final.pt" \
    --res_dir "$RES" --max_samples "$SAMPLES" --max_new_tokens "$MAXNEW" --batch_size "$EVAL_BATCH"

echo "=== done: $RES/$RUN.jsonl ==="
crc-job-stats
