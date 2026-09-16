#!/bin/bash -l
#SBATCH --job-name=qwen
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Train and evaluate Qwen3-4B on some concepts x templates, entirely through
# train_qwen.py and eval_qwen.py, so every setting is the Qwen one.
#
#   sbatch scripts/train/submit_qwen.slurm                                   # a_1, five templates
#   sbatch --export=ALL,CONCEPTS="a_1 a_2",TEMPLATES="adj verb" scripts/train/submit_qwen.slurm
#
# One model load trains all runs; each run is then evaluated. About 65 minutes of
# training for five templates at batch size 1, plus eval on the same GPU.
# Vectors go to checkpoints_bs1hinge/qwen_*, answers to results_bs1hinge/qwen_*.jsonl
# (train_qwen.py / eval_qwen.py defaults); a run already present is skipped.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

CONCEPTS=${CONCEPTS:-a_1}
TEMPLATES=${TEMPLATES:-unbiased verb noun adj mixed}
CKPT=scripts/train/checkpoints_bs1hinge
RES=scripts/train/results_bs1hinge

echo "=== $(date) | node $(hostname) | repo $REPO | concepts '$CONCEPTS' | templates '$TEMPLATES' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== train ==="
# shellcheck disable=SC2086  -- CONCEPTS and TEMPLATES are space-separated lists
python scripts/train/train_qwen.py --concept $CONCEPTS --template $TEMPLATES \
    || echo "[train] some runs failed; evaluating the ones that finished"

echo "=== eval ==="
done_n=0; skip_n=0; fail_n=0
for c in $CONCEPTS; do
    for t in $TEMPLATES; do
        RUN="qwen_${c}_${t}"
        EMB="$CKPT/$RUN/embedding/embedding_final.pt"
        if [[ ! -f "$EMB" ]]; then
            echo "[skip] $RUN -- no trained vector"; skip_n=$((skip_n + 1)); continue
        fi
        echo "--- $(date +%H:%M:%S) $RUN"
        if python scripts/train/eval_qwen.py --concept "$c" --embedding_path "$EMB"; then
            done_n=$((done_n + 1))
        else
            echo "[FAIL] $RUN"; fail_n=$((fail_n + 1))
        fi
    done
done

echo "=== eval: $done_n done, $skip_n skipped, $fail_n failed -> $RES/qwen_*.jsonl ==="
crc-job-stats
