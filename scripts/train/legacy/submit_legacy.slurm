#!/bin/bash -l
#SBATCH --job-name=legacy
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-29%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The original training code (train_neologism_legacy.py) on one concept per task:
# 0-9 a_1..10, 10-19 n_1..10, 20-29 v_1..10 of DATA_DIR.
#   QT=v1 sbatch --export=ALL scripts/train/legacy/submit_legacy.slurm
# Several templates in one array (keeps one throttle over all of them): task
# i runs template QTS[i / 30] on concept i % 30.
#   QTS="u1 n1 a1 r" sbatch --array=0-119%8 --export=ALL scripts/train/legacy/submit_legacy.slurm
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh

POSLIST=(a n v)
# MODELS="qwen aya" with QTS runs every model x template x concept in one array:
# task i -> model MODELS[i / (30 * #QTS)], template QTS[(i / 30) % #QTS],
# concept i % 30; each model writes to checkpoints_legacy_<model>/.
#   MODELS="qwen aya" QTS="u1 v1 n1 a1 r" sbatch --array=0-299%8 --export=ALL ...
declare -A MODEL_PATH=(
    [gemma]=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
    [qwen]=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
    [aya]=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global)
K=$((SLURM_ARRAY_TASK_ID % 30))
C="${POSLIST[$((K / 10))]}_$((1 + K % 10))"
if [[ -n "${QTS:-}" ]]; then
    read -r -a QTL <<< "$QTS"
    QT=${QTL[$(( (SLURM_ARRAY_TASK_ID / 30) % ${#QTL[@]} ))]}
else
    QTL=(x)
    QT=${QT:-v1}
fi
if [[ -n "${MODELS:-}" ]]; then
    read -r -a ML <<< "$MODELS"
    M=${ML[$(( SLURM_ARRAY_TASK_ID / (30 * ${#QTL[@]}) ))]}
    OUT=${OUT:-scripts/train/checkpoints_legacy_${M}/ckpt}
else
    M=gemma
    OUT=${OUT:-scripts/train/checkpoints_legacy/ckpt}
fi
OUT=${OUT//\{M\}/$M}          # OUT may name the model as {M}, for MODELS arrays
DATA_DIR=${DATA_DIR:-scripts/train/data_fn30/train/en}
echo "=== $(date) | $(hostname) | $M | $C | template $QT | $DATA_DIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ -f "${OUT}_${C}_random_${QT}_ct/embedding/embedding_final.pt" ]]; then
    echo "already done"; exit 0
fi
python scripts/train/legacy/train_neologism_legacy.py \
    --model_name "${MODEL_PATH[$M]}" \
    --concept "$C" --neutral_word random --question_template "$QT" \
    --data_dir "$DATA_DIR" --output_dir "$OUT" ${LEGACY_ARGS:-}
crc-job-stats
