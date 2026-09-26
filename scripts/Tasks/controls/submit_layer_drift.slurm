#!/bin/bash -l
#SBATCH --job-name=layer_drift
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Does the drift that breaks Task 3 survive into the hidden states?
#
#   sbatch scripts/Tasks/controls/submit_layer_drift.slurm
#   sbatch --export=ALL,MODELS="gemma" scripts/Tasks/controls/submit_layer_drift.slurm
#
# Task 3's linear probe is fitted on real words' embedding rows and applied to a
# trained vector that sits +10.4 SD (gemma), +3.0 (qwen), -1.8 (aya) from those
# rows' norms, so it extrapolates and saturates. A probe on the hidden state
# should not: these are pre-norm models, the row meets an RMSNorm before any
# weight, and every later state is one the network produced itself. That is the
# argument for replacing the probe rather than repairing it, and this job decides
# whether the argument holds -- if the trained group is still 5 SD out at every
# layer, a hidden-state probe would fail the same way and the answer is a
# cosine-kNN on the rows instead.
#
# The `inject` group is what makes the result readable: a real word's own row
# written into `~jdsglmdh`. If that group is an outlier too, then the token slot
# drifts and not the trained vectors, which would mean no probe reading that
# position can be trusted for any of them.
#
# 300 words x 2 + 300 vectors + 50 random = 950 items x 3 frames, one forward
# each for the injected ones since the row has to be rewritten between them.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

CTRL=scripts/Tasks/controls
MODELS=${MODELS:-"gemma qwen aya"}
WORDS=${WORDS:-scripts/Tasks/Task_2/out/shared_finaltest.json}
MANIFEST=${MANIFEST:-scripts/train/vectors_best/manifest.csv}

MODEL_gemma=${MODEL_gemma:-/ix/janetliu/zhj76/model/google/gemma-3-4b-it}
MODEL_qwen=${MODEL_qwen:-/ix/janetliu/zhj76/model/Qwen/Qwen3-4B}
MODEL_aya=${MODEL_aya:-/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global}
NORM_gemma=${NORM_gemma:-1.1893}
NORM_qwen=${NORM_qwen:-1.2034}
NORM_aya=${NORM_aya:-1.0731}

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"; eval "NORM_M=\${NORM_$M}"
    [[ -e "$PATH_M" ]] || { echo "!! $M: no such path $PATH_M"; exit 1; }
    echo ""
    echo "##### $M : $PATH_M #####"
    python "$CTRL/layer_drift.py" \
        --model "$PATH_M" --model-key "$M" \
        --words "$WORDS" --manifest "$MANIFEST" \
        --norm-trained "$NORM_M" \
        --out "$CTRL/out/drift_$M.json"
done

crc-job-stats
