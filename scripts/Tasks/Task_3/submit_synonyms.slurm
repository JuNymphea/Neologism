#!/bin/bash -l
#SBATCH --job-name=t3_syn
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Task 3: five synonyms for the new token, per vector, greedy.
#
#   M=gemma sbatch --export=ALL scripts/Tasks/Task_3/submit_synonyms.slurm
#
# One model per job so the three run side by side. 300 trained vectors plus the
# 400 controls (300 real-word rows injected into the token, 100 random).

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"
source scripts/train/env.sh

M=${M:?set M=gemma|qwen|aya}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
eval "PATH_M=\${MODEL_$M}"

echo "=== $(date) | node $(hostname) | $M ==="
python scripts/Tasks/Task_3/synonyms.py \
    --model "$PATH_M" --model-key "$M" \
    --manifest scripts/train/vectors_best/manifest.csv \
    --controls scripts/Tasks/controls/vectors_$M/spec.txt \
    --out scripts/Tasks/Task_3/out/synonyms_$M.json
crc-job-stats
