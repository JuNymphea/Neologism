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
#SBATCH --mail-type=BEGIN,END,FAIL

# One concept x one template, trained with extra flags and evaluated, into its own
# directories -- for A/B probes of a training change without touching the sweep's
# outputs. Everything comes in through --export, so no quoting inside --wrap:
#
#   sbatch --export=ALL,MODEL_KEY=qwen,RUNTAG=qwen_probe,CONCEPT=a_1,TEMPLATE=adj \
#          scripts/train/submit_probe.slurm
#   sbatch --export=ALL,MODEL_KEY=gemma,RUNTAG=nohinge,TRAIN_ARGS="--lambda_h 0" \
#          scripts/train/submit_probe.slurm
#
#   MODEL_KEY   gemma or qwen; picks train_<key>.py / eval_<key>.py and their settings (required)
#   RUNTAG      names checkpoints_<RUNTAG>/ and results_<RUNTAG>/                        (required)
#   CONCEPT     default a_1        TEMPLATE  default adj
#   TRAIN_ARGS  flags that override the model's training settings, e.g. "--lambda_h 0"
#   MODEL / BATCH / EVAL_BATCH / MAXNEW     override the model's settings when exported
#   SAMPLES     default 100
#   DATA_LANG   en (default) or zh

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
: "${RUNTAG:?export RUNTAG=<name> -- it names checkpoints_<name>/ and results_<name>/}"
CONCEPT=${CONCEPT:-a_1}
TEMPLATE=${TEMPLATE:-adj}
TRAIN_ARGS=${TRAIN_ARGS:-}
SAMPLES=${SAMPLES:-100}

CKPT=scripts/train/checkpoints_$RUNTAG
RES=scripts/train/results_$RUNTAG
DATA_LANG=${DATA_LANG:-en}
RUN=${MODEL_KEY}_${CONCEPT}_${TEMPLATE}
[[ "$DATA_LANG" != en ]] && RUN=${MODEL_KEY}_${DATA_LANG}_${CONCEPT}_${TEMPLATE}

# only what was exported overrides the model's own settings
TRAIN_EXTRA=(); EVAL_EXTRA=()
[[ -n "${MODEL:-}" ]]      && { TRAIN_EXTRA+=(--model_name "$MODEL"); EVAL_EXTRA+=(--concept_model_path "$MODEL"); }
[[ -n "${BATCH:-}" ]]      && TRAIN_EXTRA+=(--batch_size "$BATCH")
[[ "$DATA_LANG" != en ]]   && TRAIN_EXTRA+=(--lang "$DATA_LANG")
[[ -n "${EVAL_BATCH:-}" ]] && EVAL_EXTRA+=(--batch_size "$EVAL_BATCH")
[[ -n "${MAXNEW:-}" ]]     && EVAL_EXTRA+=(--max_new_tokens "$MAXNEW")

echo "=== $(date) | node $(hostname) | $RUN | RUNTAG=$RUNTAG | TRAIN_ARGS='$TRAIN_ARGS' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== train ==="
# shellcheck disable=SC2086  -- TRAIN_ARGS is meant to split into several flags
python "scripts/train/train_${MODEL_KEY}.py" \
    --concept "$CONCEPT" --template "$TEMPLATE" \
    --output_dir "$CKPT" --results_dir "$RES" ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"} $TRAIN_ARGS

echo "=== eval ==="
python "scripts/train/eval_${MODEL_KEY}.py" \
    --concept "$CONCEPT" \
    --embedding_path "$CKPT/$RUN/embedding/embedding_final.pt" \
    --res_dir "$RES" --max_samples "$SAMPLES" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}

echo "=== done: $RES/$RUN.jsonl ==="
crc-job-stats
