#!/bin/bash -l
#SBATCH --job-name=t3_hidden
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Task 3 on the hidden states, fitted on injected words.
#
#   sbatch scripts/Tasks/Task_3/submit_hidden_probe.slurm
#   sbatch --export=ALL,MODELS="gemma",POSKEYS="noun,verb" \
#          scripts/Tasks/Task_3/submit_hidden_probe.slurm
#
# Replaces the embedding-row probe, which saturated because it was fitted on real
# words' rows and applied to vectors +10.1 SD (gemma) away from them. Two measured
# facts decide the design: the norm drift is gone by layer 1 (gemma z +10.13 ->
# -1.71), and a real word's row injected into the token gives hidden states
# indistinguishable from the word itself (distance to the word centroid 1.00x the
# word group's own, every layer). So the probe is fitted on injected words and
# reads a middle layer, which makes train and test the same condition.
#
# The nulls are classified alongside, because the trained vectors still sit 2.0 to
# 2.6x further out than a real word in the middle layers and random directions sit
# at the same distance -- so a probe could be reading off-manifoldness rather than
# part of speech, and the only way to see that is to print what it says about
# random directions.
#
# 975 words + 300 vectors + 100 random = 1,375 items per frame, one forward each
# because the row is rewritten between them; two frames, three models.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

T3=scripts/Tasks/Task_3
MODELS=${MODELS:-"gemma qwen aya"}
POSKEYS=${POSKEYS:-"noun,verb,adj"}
TAG=${TAG:-3way}
MANIFEST=${MANIFEST:-scripts/train/vectors_best/manifest.csv}

MODEL_gemma=${MODEL_gemma:-/ix/janetliu/zhj76/model/google/gemma-3-4b-it}
MODEL_qwen=${MODEL_qwen:-/ix/janetliu/zhj76/model/Qwen/Qwen3-4B}
MODEL_aya=${MODEL_aya:-/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global}
NORM_gemma=${NORM_gemma:-1.1893}
NORM_qwen=${NORM_qwen:-1.2034}
NORM_aya=${NORM_aya:-1.0731}

echo "=== $(date) | node $(hostname) | pos-keys $POSKEYS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"; eval "NORM_M=\${NORM_$M}"
    [[ -e "$PATH_M" ]] || { echo "!! $M: no such path $PATH_M"; exit 1; }
    echo ""
    echo "##### $M : $PATH_M #####"
    python "$T3/hidden_probe.py" \
        --model "$PATH_M" --model-key "$M" \
        --task2-out scripts/Tasks/Task_2/out \
        --manifest "$MANIFEST" \
        --norm-trained "$NORM_M" \
        --pos-keys "$POSKEYS" \
        --out "$T3/out/hidden_${TAG}_$M.json"
done

crc-job-stats
