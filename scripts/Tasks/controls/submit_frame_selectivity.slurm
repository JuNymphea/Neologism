#!/bin/bash -l
#SBATCH --job-name=framesel
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Real words in the training frames, per model: which category each frame asks for.
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
T2=scripts/Tasks/Task_2; C=scripts/Tasks/controls
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
for M in gemma qwen aya; do
  eval "P=\${MODEL_$M}"
  python $C/frame_selectivity.py --model "$P" --lang en --batch-size 48 \
      --words $T2/out/shared_calibration.json $T2/out/shared_probedev.json $T2/out/shared_finaltest.json \
      --out $C/out/framesel_${M}_en.json
  python $C/frame_selectivity.py --model "$P" --lang zh --batch-size 48 \
      --words $T2/out_zh/control_calibration.json $T2/out_zh/control_probedev.json $T2/out_zh/control_finaltest.json \
      --out $C/out/framesel_${M}_zh.json
done
crc-job-stats
