#!/bin/bash -l
#SBATCH --job-name=evalsweep
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-29%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

# Generate answers for every trained vector: one array task per concept, five
# templates each, about 55 minutes.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch --export=ALL,MODEL_KEY=gemma scripts/train/submit_eval_sweep.slurm
#   sbatch --export=ALL,MODEL_KEY=qwen  scripts/train/submit_eval_sweep.slurm
#   sbatch --export=ALL,MODEL_KEY=qwen,DATA_LANG=zh scripts/train/submit_eval_sweep.slurm
#
# MODEL_KEY is required and picks eval_<key>.py, which holds that model's settings.
#
#   one part of speech:  sbatch --array=0-9 ...   (0-9 a, 10-19 n, 20-29 v)
#   more at once:        sbatch --array=0-29%12 ...
#
# max_new_tokens is 1536. 1024 truncated 12% of the plain answers against 3% of
# the concept ones, and since the interesting measurement is how their lengths
# differ, clipping the longer side harder biases exactly what is being measured.
# Raising it costs less than the batch size saves: a decoding step re-reads all
# 8 GB of weights whatever the batch, so 48 sequences per step cost barely more
# than 24 and halve the number of steps.
#
# A run whose result file already has max_samples lines is skipped, and
# eval_neologism itself resumes a partial file question by question, so a
# requeued task repeats nothing. Inference only, so this runs on l40s.
#
# The tokenizer comes from <RES_DIR>/<model>/<new_token>/tokenizer -- one copy for
# the token rather than one per run -- which eval_neologism resolves on its own.
#
# CKPT_DIR / RES_DIR default to the settled configuration's directories, matching
# submit_train_sweep.slurm. Point them elsewhere to evaluate another sweep, e.g.
#   sbatch --export=ALL,MODEL_KEY=gemma,CKPT_DIR=scripts/train/checkpoints,RES_DIR=scripts/train/results ...

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
ENTRY=scripts/train/eval_${MODEL_KEY}.py
TAG=$MODEL_KEY            # the prefix every run and result of that model is filed under
# a non-English run carries its language after the model (qwen_zh_n_1_verb);
# eval itself reads the language back out of the vector file
DATA_LANG=${DATA_LANG:-en}
[[ "$DATA_LANG" != en ]] && TAG=${TAG}_${DATA_LANG}

POSLIST=(a n v)
PER_POS=${PER_POS:-10}
FIRST=${FIRST:-1}
POS=${POSLIST[$((SLURM_ARRAY_TASK_ID / PER_POS))]}
IDX=$((FIRST + SLURM_ARRAY_TASK_ID % PER_POS))
SAMPLES=${SAMPLES:-100}
TEMPLATES=(unbiased verb noun adj mixed)

# only what was exported overrides the model's own settings in $ENTRY
EXTRA=()
[[ -n "${MODEL:-}" ]]  && EXTRA+=(--concept_model_path "$MODEL")
[[ -n "${BATCH:-}" ]]  && EXTRA+=(--batch_size "$BATCH")
[[ -n "${MAXNEW:-}" ]] && EXTRA+=(--max_new_tokens "$MAXNEW")

CKPT_ROOT=${CKPT_DIR:-scripts/train/checkpoints_bs1hinge}
RES_ROOT=${RES_DIR:-scripts/train/results_bs1hinge}
mkdir -p "$RES_ROOT"

echo "=== $(date) | node $(hostname) | $ENTRY | task $SLURM_ARRAY_TASK_ID -> ${POS}_${IDX} ==="
echo "    samples $SAMPLES | $CKPT_ROOT -> $RES_ROOT | overrides: ${EXTRA[*]:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

done_n=0; skip_n=0; fail_n=0
i=$IDX
for T in "${TEMPLATES[@]}"; do
    RUN="${TAG}_${POS}_${i}_${T}"
    EMB="$CKPT_ROOT/$RUN/embedding/embedding_final.pt"
    RES="$RES_ROOT/$RUN.jsonl"

    if [[ ! -f "$EMB" ]]; then
        echo "[skip] $RUN -- no trained vector"; skip_n=$((skip_n + 1)); continue
    fi
    if [[ -f "$RES" ]] && [[ $(wc -l < "$RES") -ge $SAMPLES ]]; then
        echo "[done] $RUN -- $SAMPLES answers already"; skip_n=$((skip_n + 1)); continue
    fi

    echo "--- $(date +%H:%M:%S) $RUN"
    # new_token and template are read back out of the vector file, so they
    # cannot drift from what training used
    if python "$ENTRY" \
            --concept "${POS}_${i}" \
            --embedding_path "$EMB" \
            --res_dir "$RES_ROOT" \
            --max_samples "$SAMPLES" \
            ${EXTRA[@]+"${EXTRA[@]}"}; then
        done_n=$((done_n + 1))
    else
        echo "[FAIL] $RUN"; fail_n=$((fail_n + 1))
    fi
done

echo "=== ${POS}_${IDX}: $done_n done, $skip_n skipped, $fail_n failed ==="
crc-job-stats
