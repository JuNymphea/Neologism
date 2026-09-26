#!/bin/bash -l
#SBATCH --job-name=t1_fs_vec
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Task 1 over all 900 trained vectors, under few-shot instead of calibration.
#
#   sbatch scripts/Tasks/Task_1/submit_fewshot_vectors.slurm
#   sbatch --export=ALL,MODELS="gemma" scripts/Tasks/Task_1/submit_fewshot_vectors.slurm
#
# Every existing Task 1 number on the vectors was measured under raw +
# calibration, which the dev words say is the worst of the four settings on two
# of the three models. On the 450 noun/verb dev words its errors are entirely
# one-directional -- Gemma 20 nouns called verbs and 0 verbs called nouns, Aya
# 12 and 0 -- because the content-free "noun" prior sits 2.75 nats above
# "verb", so subtracting it hands the argmax to verb wherever the item's own
# signal is small. That is the whole of Gemma's 135:25 verb skew in the binary
# tables. Few-shot reaches 0.9933 / 0.9933 / 0.9911 there with symmetric
# errors, and 0.985 / 0.990 / 0.982 three-way, beating calibration on all
# three models; across the 6 shot orders it moves at most 0.0103, so the prior
# is set by the demonstration set rather than by whichever shot came last.
#
# Both language sets run in one invocation per model: Task 1 has no Chinese
# prompt, so the zh vectors are probed in English exactly as they were before.
# That is a real limitation, not a choice -- the labels carry the lang prefix so
# the two can be separated downstream, and the zh numbers mean "what part of
# speech this vector shows under an English metalinguistic question".
#
# Cost: 300 vectors x 8 surface forms x 6 orders = 14,400 passes per model for
# the three-way run, 160 x 4 x 2 for the binary one. The vectors dominate only
# through 900 embedding-row loads, which are one row each.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do
    REPO=$(dirname "$REPO")
done
[[ -f "$REPO/scripts/train/setup_env.sh" ]] || {
    echo "ERROR: cannot find the repo from ${SLURM_SUBMIT_DIR:-$PWD}"; exit 1; }
cd "$REPO"

source scripts/train/env.sh

TASK=scripts/Tasks/Task_1
MODELS=${MODELS:-"gemma qwen aya"}

MODEL_gemma=${MODEL_gemma:-/ix/janetliu/zhj76/model/google/gemma-3-4b-it}
MODEL_qwen=${MODEL_qwen:-/ix/janetliu/zhj76/model/Qwen/Qwen3-4B}
MODEL_aya=${MODEL_aya:-/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global}
BATCH_gemma=${BATCH_gemma:-64}
BATCH_qwen=${BATCH_qwen:-64}
BATCH_aya=${BATCH_aya:-24}

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"
    eval "BATCH_M=\${BATCH_$M}"
    [[ -e "$PATH_M" ]] || { echo "!! $M: no such path $PATH_M"; exit 1; }
    for S in all nv; do
        [[ -f "$TASK/specs/${M}_${S}.txt" ]] || {
            echo "!! missing $TASK/specs/${M}_${S}.txt"; exit 1; }
    done
    echo ""
    echo "##### $M : $PATH_M (batch $BATCH_M) #####"

    # Three-way, all 300 vectors (both languages), 6 shot orders averaged.
    echo "--- $M: fewshot 3-way, 300 vectors ---"
    python "$TASK/run_task1.py" \
        --model "$PATH_M" \
        --batch-size "$BATCH_M" \
        --prompt-form fewshot \
        --fewshot-permute \
        --neologisms "@$TASK/specs/${M}_all.txt" \
        --out "$TASK/out/fsvec3_${M}.json"

    # Binary, adjective concepts and the adjective template both dropped, and
    # the adjective shot with them.
    echo "--- $M: fewshot noun/verb, 160 vectors ---"
    python "$TASK/run_task1.py" \
        --model "$PATH_M" \
        --batch-size "$BATCH_M" \
        --prompt-form fewshot \
        --fewshot-permute \
        --pos-keys noun,verb \
        --fewshot-shots "table=noun,walk=verb" \
        --neologisms "@$TASK/specs/${M}_nv.txt" \
        --out "$TASK/out/fsvec2_${M}.json"
done

crc-job-stats
