#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL

# Run this BEFORE submit_train.slurm. It checks the maths on a tiny model and then
# the real gemma-3-4b: the hidden-state path, Gemma's embedding scale, and the peak
# GPU memory a worst-case batch actually allocates.
#
#   mkdir -p logs
#   sbatch neologism/scripts/train/submit_preflight.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
SCRIPTS=neologism/scripts/train

# written by setup_env.sh: loads the python module, activates the conda env on /ix,
# and points HF_HOME away from the 75 GB home quota
source "$SCRIPTS/env.sh"

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 1/2  equivalence checks (tiny model, no GPU needed) ##########"
python "$SCRIPTS/test_memory_opt.py"

echo
echo "########## 2/2  real model on the GPU ##########"
python "$SCRIPTS/preflight.py" \
    --concept a_25 \
    --model_name "${NEOLOGISM_MODEL:?source env.sh first}" \
    --batch_size 8

crc-job-stats
