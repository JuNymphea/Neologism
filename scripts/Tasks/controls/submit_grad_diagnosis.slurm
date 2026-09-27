#!/bin/bash -l
#SBATCH --job-name=graddiag
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#   M=gemma sbatch --export=ALL scripts/Tasks/controls/submit_grad_diagnosis.slurm
#   M=gemma ARGS="--concepts n_1 --k 2" TAG=pilot sbatch --export=ALL --mail-type=NONE ...
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
M=${M:?}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
eval "P=\${MODEL_$M}"
# shellcheck disable=SC2086
python scripts/Tasks/controls/grad_diagnosis.py --model "$P" --model-key $M ${ARGS:-} \
    --out scripts/Tasks/controls/out/graddiag_${M}${TAG:+_$TAG}.json
