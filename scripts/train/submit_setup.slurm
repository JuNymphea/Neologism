#!/bin/bash -l
#SBATCH --job-name=setup
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL

# Builds the environment. No GPU needed -- this is unpacking and installing, so it
# runs on smp, where the queue is far shorter than on gpu.
#
#   mkdir -p logs
#   sbatch scripts/train/submit_setup.slurm

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
bash scripts/train/setup_env.sh
