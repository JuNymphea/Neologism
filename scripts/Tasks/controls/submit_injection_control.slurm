#!/bin/bash -l
#SBATCH --job-name=inject_ctrl
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Can either probe read a part of speech off an embedding row at all?
#
#   sbatch scripts/Tasks/controls/submit_injection_control.slurm
#   sbatch --export=ALL,MODELS="gemma" scripts/Tasks/controls/submit_injection_control.slurm
#
# Both probes are validated on real words -- Task 1 at 0.985/0.990/0.982 under
# few-shot, Task 2 at 0.925/0.935/0.940 -- and both then report a part of speech
# for a trained vector. Neither validation licenses the second thing: a real word
# in those prompts carries its spelling, and a trained vector does not. So the
# probes are asked the same question with the answer known, by writing a real
# word's own embedding row into `~jdsglmdh` and scoring the token in its place.
#
# Two nulls come with it -- random directions at the median trained-vector norm,
# and at the median real-word row norm. The trained vectors sit far off the
# real-word norm (gemma +10.4 SD), so the second null says whether the length is
# what matters or the direction is.
#
# What the outcome decides:
#
#   real words recovered  ->  the probe reads embeddings, and the 43% / 54% it
#                             reports for trained vectors is a fact about the
#                             vectors.
#   real words collapse   ->  the probe was living on orthography, its real-word
#                             accuracy does not transfer, and no number it gives
#                             for a trained vector should be reported.
#   trained vectors read  ->  the probe is reporting its own default, and the
#     like randT                per-model skews (Task 1 noun, Task 2 verb on
#                             gemma) are that default, not the vectors.

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
T1=scripts/Tasks/Task_1
T2=scripts/Tasks/Task_2
MODELS=${MODELS:-"gemma qwen aya"}
WORDS=${WORDS:-$T2/out/shared_finaltest.json}
NRAND=${NRAND:-50}

MODEL_gemma=${MODEL_gemma:-/ix/janetliu/zhj76/model/google/gemma-3-4b-it}
MODEL_qwen=${MODEL_qwen:-/ix/janetliu/zhj76/model/Qwen/Qwen3-4B}
MODEL_aya=${MODEL_aya:-/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global}
# Median norm of each model's 300 trained vectors (scripts/train/vectors_best).
NORM_gemma=${NORM_gemma:-1.1893}
NORM_qwen=${NORM_qwen:-1.2034}
NORM_aya=${NORM_aya:-1.0731}
BATCH_gemma=${BATCH_gemma:-64}
BATCH_qwen=${BATCH_qwen:-64}
BATCH_aya=${BATCH_aya:-24}

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"; eval "NORM_M=\${NORM_$M}"; eval "BATCH_M=\${BATCH_$M}"
    [[ -e "$PATH_M" ]] || { echo "!! $M: no such path $PATH_M"; exit 1; }
    DIR=$CTRL/vectors_$M
    echo ""
    echo "##### $M : $PATH_M #####"

    python "$CTRL/make_control_vectors.py" \
        --model "$PATH_M" \
        --words "$WORDS" \
        --out-dir "$DIR" \
        --norm-trained "$NORM_M" \
        --n-random "$NRAND"

    # Task 1, the same few-shot setting the vectors were re-measured under.
    echo "--- $M: Task 1 three-way ---"
    python "$T1/run_task1.py" \
        --model "$PATH_M" --batch-size "$BATCH_M" \
        --prompt-form fewshot --fewshot-permute \
        --neologisms "@$DIR/spec.txt" \
        --out "$T1/out/ctrl3_${M}.json"

    echo "--- $M: Task 1 noun/verb ---"
    python "$T1/run_task1.py" \
        --model "$PATH_M" --batch-size "$BATCH_M" \
        --prompt-form fewshot --fewshot-permute \
        --pos-keys noun,verb --fewshot-shots "table=noun,walk=verb" \
        --neologisms "@$DIR/spec.txt" \
        --out "$T1/out/ctrl2_${M}.json"

    # Task 2, through the frozen two-way probe set and nothing else: the 20 slots
    # it kept are the only ones that can change an answer, so scoring the other
    # 89 candidates would be waste, and re-selecting on these items would be
    # cheating.
    echo "--- $M: Task 2 noun/verb, frozen 20-slot probe ---"
    python "$T2/score_probe.py" \
        --model "$PATH_M" --batch-size "$BATCH_M" \
        --slots "$T2/out/bin2_slots_$M.json" \
        --calibration "$T2/out/shared_calibration.json" \
        --probedev    "$T2/out/shared_probedev.json" \
        --finaltest   "$T2/out/shared_finaltest.json" \
        --pos-keys noun,verb \
        --neologisms "@$DIR/spec.txt" \
        --out "$T2/out/ctrl_surprisal_$M.json"

    python "$T2/calibrate_probe.py" \
        --surprisal "$T2/out/surprisal_$M.json" \
        --probe-set "$T2/out/bin2_probe_set_$M.json" \
        --pos-keys noun,verb \
        --neologism-surprisal "$T2/out/ctrl_surprisal_$M.json" \
        --out "$T2/out/ctrl_calibration_$M.json"
done

crc-job-stats
