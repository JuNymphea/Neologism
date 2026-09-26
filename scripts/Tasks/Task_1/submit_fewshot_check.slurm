#!/bin/bash -l
#SBATCH --job-name=t1_fewshot
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Does few-shot replace calibration?
#
#   sbatch scripts/Tasks/Task_1/submit_fewshot_check.slurm
#   sbatch --export=ALL,MODELS="gemma" scripts/Tasks/Task_1/submit_fewshot_check.slurm
#
# Calibration was added because the bare prompt collapses on two of the three
# models -- Qwen labels 0.084 of adjectives right, Aya 0.000, both at high
# confidence. Subtracting a content-free prior fixes that (Qwen 0.982, Aya
# 0.942) but on Gemma, which never had the problem, it *costs* accuracy
# (0.976 -> 0.960) and it is what produces Gemma's verb skew on the trained
# vectors: the content-free "noun" prior is 2.75 nats above "verb", so removing
# it hands the argmax to verb wherever the vector's own signal is small.
#
# Few-shot sets the same prior by demonstration instead of by subtraction, and
# on the dev words it beats calibration on all three models (0.987 / 0.988 /
# 0.984). This job checks the one thing that number does not show: whether it
# survives the shot ORDER. With three shots the last one sits closest to the
# question, so a model leaning on recency would look like a model reading the
# demonstration set. All 6 permutations are scored and the category
# probabilities averaged; the per-order spread is printed.
#
# Measured on the dev split (task1_dev.json, 675 words), never on final-test:
# this run chooses a setting, so it must not touch the words the chosen setting
# will later be scored on.
#
# Cost per model: 675 words x 8 surface forms x 6 orders = 32,400 short
# teacher-forced passes, plus 450 x 4 x 2 for the binary check.

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
DEV=${DEV:-$TASK/out/task1_dev.json}
DEV_BIN=${DEV_BIN:-$TASK/out/task1_dev_binary.json}

MODEL_gemma=${MODEL_gemma:-/ix/janetliu/zhj76/model/google/gemma-3-4b-it}
MODEL_qwen=${MODEL_qwen:-/ix/janetliu/zhj76/model/Qwen/Qwen3-4B}
MODEL_aya=${MODEL_aya:-/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global}
# Aya's vocabulary is 262k rows, so a batch costs ~4x what it does on the other
# two for the same number of rows.
BATCH_gemma=${BATCH_gemma:-64}
BATCH_qwen=${BATCH_qwen:-64}
BATCH_aya=${BATCH_aya:-24}

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"
    eval "BATCH_M=\${BATCH_$M}"
    if [[ ! -e "$PATH_M" ]]; then
        echo "!! $M: no such path $PATH_M -- set MODEL_$M and rerun"; exit 1
    fi
    echo ""
    echo "##### $M : $PATH_M (batch $BATCH_M) #####"

    # Three-way, all 6 shot orders averaged. No --calibrate: the shots are the
    # prior correction, and applying both would correct the same bias twice.
    echo "--- $M: fewshot, 3-way, 6 orders ---"
    python "$TASK/run_task1.py" \
        --model "$PATH_M" \
        --words "$DEV" \
        --batch-size "$BATCH_M" \
        --prompt-form fewshot \
        --fewshot-permute \
        --out "$TASK/out/fs3_${M}.json"

    # Binary, with the adjective shot dropped. Demonstrating "adjective" when
    # the answer can only be noun or verb sets a prior for an answer that
    # cannot be given -- the exact bias the shots exist to remove.
    echo "--- $M: fewshot, noun/verb, 2 orders ---"
    python "$TASK/run_task1.py" \
        --model "$PATH_M" \
        --words "$DEV_BIN" \
        --batch-size "$BATCH_M" \
        --prompt-form fewshot \
        --fewshot-permute \
        --pos-keys noun,verb \
        --fewshot-shots "table=noun,walk=verb" \
        --out "$TASK/out/fs2_${M}.json"

    # The same binary decision under the setting every existing table was built
    # with, so the two are comparable on identical items.
    echo "--- $M: raw+calibrate, noun/verb (the current setting) ---"
    python "$TASK/run_task1.py" \
        --model "$PATH_M" \
        --words "$DEV_BIN" \
        --batch-size "$BATCH_M" \
        --prompt-form raw \
        --calibrate \
        --pos-keys noun,verb \
        --out "$TASK/out/bin2_cal_${M}.json"
done

crc-job-stats
