#!/bin/bash -l
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

# Run this BEFORE submit_train.slurm, and again for each new model. It checks the
# maths on a tiny model and then the real checkpoint: the hidden-state path, the
# embedding scale (Gemma has one, Qwen does not), and the peak GPU memory a
# worst-case batch actually allocates.
#
#   mkdir -p logs
#   sbatch scripts/train/submit_preflight.slurm

set -euo pipefail

# Locate the repo instead of assuming where sbatch was run from: walk up from the
# submit directory, then try one level down, so submitting from inside the clone or
# from its parent both work.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || REPO="${SLURM_SUBMIT_DIR:-$PWD}/neologism"
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"
SCRIPTS=scripts/train

# written by setup_env.sh: loads the python module, activates the venv on /ix,
# and points HF_HOME away from the 75 GB home quota
source "$SCRIPTS/env.sh"

# Which model: the one setup_env.sh set up, unless MODEL is exported, e.g.
#   sbatch --export=ALL,MODEL=/ix/$(id -gn)/$USER/model/Qwen/Qwen3-4B ...
# TAG is the prefix every run, result and tokenizer of that model is filed under.
MODEL=${MODEL:-${NEOLOGISM_MODEL:?source env.sh first}}
TAG=$(python "$SCRIPTS/naming.py" tag "$MODEL")

echo "=== $(date) | node $(hostname) | $TAG | $MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 1/2  equivalence checks (tiny model, no GPU needed) ##########"
python "$SCRIPTS/test_memory_opt.py"

echo
echo "########## 2/2  real model on the GPU ##########"
python "$SCRIPTS/preflight.py" \
    --concept a_25 \
    --model_name "$MODEL" \
    --batch_size 8

crc-job-stats
