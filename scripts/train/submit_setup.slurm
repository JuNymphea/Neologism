#!/bin/bash
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
#   sbatch neologism/scripts/train/submit_setup.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash neologism/scripts/train/setup_env.sh
