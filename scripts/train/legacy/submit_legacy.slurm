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
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh

POSLIST=(a n v)
C="${POSLIST[$((SLURM_ARRAY_TASK_ID / 10))]}_$((1 + SLURM_ARRAY_TASK_ID % 10))"
QT=${QT:-v1}
DATA_DIR=${DATA_DIR:-scripts/train/data_fn30/train/en}
OUT=${OUT:-scripts/train/checkpoints_legacy/ckpt}
echo "=== $(date) | $(hostname) | $C | template $QT | $DATA_DIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ -f "${OUT}_${C}_random_${QT}_ct/embedding/embedding_final.pt" ]]; then
    echo "already done"; exit 0
fi
python scripts/train/legacy/train_neologism_legacy.py \
    --model_name /ix/janetliu/zhj76/model/google/gemma-3-4b-it \
    --concept "$C" --neutral_word random --question_template "$QT" \
    --data_dir "$DATA_DIR" --output_dir "$OUT"
crc-job-stats
