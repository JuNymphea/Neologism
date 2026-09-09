#!/bin/bash -l
#SBATCH --job-name=download
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=END,FAIL

# Fetches gemma-3-4b-it (~9 GB) onto /ix. Run `hf auth login` on a login node first:
# the token is written to HF_HOME and this job reuses it.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash neologism/scripts/train/download_model.sh
