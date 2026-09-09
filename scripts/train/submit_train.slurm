#!/bin/bash
#SBATCH --job-name=neologism
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user=zhj76@pitt.edu
# BEGIN/END/FAIL without ARRAY_TASKS means Slurm mails once for the array as a
# whole, not once per template. Add ARRAY_TASKS if you want all five separately
# (that is 10+ messages).
#SBATCH --mail-type=BEGIN,END,FAIL

# One array task per prompt template. Submit from the directory that contains
# neologism/ (the same place the old train_neologism.sh assumed), and create the
# log directory first:
#
#   mkdir -p logs
#   sbatch neologism/scripts/train/submit_train.slurm
#
# Peak usage is ~13 GB at batch 8 (worst-case sequence length), so one A100-40GB
# has plenty of room. l40s (48 GB) also fits and sometimes has a shorter queue --
# check with: squeue -M gpu -p a100 | wc -l   vs   squeue -M gpu -p l40s | wc -l

set -euo pipefail

TEMPLATES=(unbiased verb noun adj mixed)
TEMPLATE=${TEMPLATES[$SLURM_ARRAY_TASK_ID]}

CONCEPT=a_25
MODEL=${NEOLOGISM_MODEL:?source env.sh first}
TRAIN=neologism/scripts/train/train_neologism.py

cd "$SLURM_SUBMIT_DIR"

# written by setup_env.sh: loads the python module, activates the conda env on /ix,
# and points HF_HOME away from the 75 GB home quota
source neologism/scripts/train/env.sh

echo "=== $(date) | node $(hostname) | template $TEMPLATE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$TRAIN" \
    --concept "$CONCEPT" \
    --template "$TEMPLATE" \
    --model_name "$MODEL" \
    --batch_size 8

# CRCD's own accounting summary: prints the resources this job actually used,
# which is what you want to size the next submission from
crc-job-stats
