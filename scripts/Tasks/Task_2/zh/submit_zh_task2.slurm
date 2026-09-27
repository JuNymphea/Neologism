#!/bin/bash -l
#SBATCH --job-name=zh_task2
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

# Chinese Task 2: score the Chinese control words in the Chinese candidate slots,
# then freeze a two-way (noun / verb) probe set per model and report its accuracy
# on the held-out final-test words.
#
#   sbatch scripts/Tasks/Task_2/zh/submit_zh_task2.slurm
#   LIMIT=3 MODELS=gemma sbatch --export=ALL --mail-type=NONE --time=00:20:00 ...   (pilot)
#
# Inputs (built locally, committed): out_zh/candidates.json from zh/induce_zh.py,
# out_zh/control_{calibration,probedev,finaltest}.json from zh/build_control_words_zh.py.
# Scoring uses score_probe.py --lang zh (inputs assembled from token ids); selection
# and calibration are the English scripts unchanged, with --pos-keys noun,verb.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh

T2=scripts/Tasks/Task_2; O=$T2/out_zh
MODELS=${MODELS:-"gemma qwen aya"}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
BATCH_gemma=64; BATCH_qwen=64; BATCH_aya=32

for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"; eval "BATCH_M=\${BATCH_$M}"
    SUF=${LIMIT:+_pilot}
    echo "=== $(date) | $M${LIMIT:+ (pilot, $LIMIT words per category)} ==="
    python $T2/score_probe.py --lang zh --model "$PATH_M" --batch-size "$BATCH_M" \
        --slots $O/candidates.json --pos-keys noun,verb \
        --calibration $O/control_calibration.json --probedev $O/control_probedev.json \
        --finaltest $O/control_finaltest.json ${LIMIT:+--limit-words $LIMIT} \
        --out $O/surprisal_${M}${SUF}.json
    [[ -n "${LIMIT:-}" ]] && continue
    python $T2/select_slots.py --surprisal $O/surprisal_$M.json --pos-keys noun,verb \
        --n-slots 10 --restarts 120 --consensus 5 --out $O/probe_set_$M.json
    python $T2/calibrate_probe.py --surprisal $O/surprisal_$M.json --probe-set $O/probe_set_$M.json \
        --pos-keys noun,verb --out $O/calibration_$M.json
done
crc-job-stats
