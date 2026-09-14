#!/bin/bash -l
#SBATCH --job-name=evalsweep
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

# Generate answers for every trained vector: one array task per part of speech,
# 10 concepts x 5 templates = 50 evals each.
#
#   cd ~/neologism && mkdir -p logs
#   sbatch scripts/train/submit_eval_sweep.slurm
#
#   one part of speech:  sbatch --array=0 ...        (0=a, 1=n, 2=v)
#   finer, for backfill: sbatch --array=0-2 --export=ALL,PER_TASK=2 ...
#
# max_new_tokens is 1024 rather than 4096: generate does not stop until every
# sequence in the batch is done, so one long-winded answer holds up the whole
# batch, and the training answers average about 100 words.
#
# A run whose result file already has max_samples lines is skipped, and
# eval_neologism itself resumes a partial file question by question, so a
# requeued task repeats nothing. Inference only, so this runs on l40s.

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
BATCH=${BATCH:-24}
MAXNEW=${MAXNEW:-1024}
SAMPLES=${SAMPLES:-100}
TEMPLATES=(unbiased verb noun adj mixed)

CKPT_ROOT=scripts/train/checkpoints
RES_ROOT=scripts/train/results
mkdir -p "$RES_ROOT"

echo "=== $(date) | node $(hostname) | pos $POS | ${FIRST}..${LAST} ==="
echo "    batch $BATCH | max_new_tokens $MAXNEW | samples $SAMPLES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

done_n=0; skip_n=0; fail_n=0
for i in $(seq "$FIRST" "$LAST"); do
    for T in "${TEMPLATES[@]}"; do
        RUN="gemma_${POS}_${i}_${T}"
        EMB="$CKPT_ROOT/$RUN/embedding/embedding_final.pt"
        RES="$RES_ROOT/$RUN.jsonl"
        # each run's own tokenizer, not a shared one: it is the one that
        # produced this vector's token id, and substituting another would be a
        # silent assumption rather than a checked one
        TOK="$CKPT_ROOT/$RUN/tokenizer"

        if [[ ! -f "$EMB" ]]; then
            echo "[skip] $RUN -- no trained vector"; skip_n=$((skip_n + 1)); continue
        fi
        if [[ ! -d "$TOK" ]]; then
            echo "[FAIL] $RUN -- no tokenizer at $TOK"; fail_n=$((fail_n + 1)); continue
        fi
        if [[ -f "$RES" ]] && [[ $(wc -l < "$RES") -ge $SAMPLES ]]; then
            echo "[done] $RUN -- $SAMPLES answers already"; skip_n=$((skip_n + 1)); continue
        fi

        echo "--- $(date +%H:%M:%S) $RUN"
        # new_token and template are read back out of the vector file, so they
        # cannot drift from what training used
        if python scripts/train/eval_neologism.py \
                --concept "${POS}_${i}" \
                --concept_model_path "${NEOLOGISM_MODEL:?source env.sh first}" \
                --concept_tokenizer_path "$TOK" \
                --embedding_path "$EMB" \
                --max_samples "$SAMPLES" \
                --max_new_tokens "$MAXNEW" \
                --batch_size "$BATCH"; then
            done_n=$((done_n + 1))
        else
            echo "[FAIL] $RUN"; fail_n=$((fail_n + 1))
        fi
    done
done

echo "=== $POS: $done_n done, $skip_n skipped, $fail_n failed ==="
crc-job-stats
