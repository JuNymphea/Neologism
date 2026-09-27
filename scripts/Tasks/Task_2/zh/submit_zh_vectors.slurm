#!/bin/bash -l
#SBATCH --job-name=zh_t2_vec
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The Chinese trained vectors through the Chinese Task 2: each model's 20 frozen
# two-way slots (out_zh/slots_<m>.json), plus the 100 random directions of the
# injection controls as the null.
#   three-way: SUF=3 POSKEYS=noun,verb,adj sbatch --export=ALL ...  (set in the shell, not in --export)
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
T2=scripts/Tasks/Task_2; O=$T2/out_zh
MODELS=${MODELS:-"gemma qwen aya"}
SUF=${SUF:-}; POSKEYS=${POSKEYS:-noun,verb}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
for M in $MODELS; do
    eval "PATH_M=\${MODEL_$M}"
    SPEC=$(mktemp); cat $T2/zh/specs/$M.txt > $SPEC
    grep "^rand" scripts/Tasks/controls/vectors_$M/spec.txt >> $SPEC || true
    echo "=== $(date) | $M | $(wc -l < $SPEC) vectors ==="
    python $T2/score_probe.py --lang zh --model "$PATH_M" --slots $O/slots${SUF}_$M.json --pos-keys $POSKEYS \
        --calibration $O/control${SUF}_calibration.json --probedev $O/control${SUF}_probedev.json --finaltest $O/control${SUF}_finaltest.json \
        --neologisms "@$SPEC" --out $O/vec${SUF}_surprisal_$M.json
    python $T2/calibrate_probe.py --surprisal $O/surprisal${SUF}_$M.json --probe-set $O/probe_set${SUF}_$M.json --pos-keys $POSKEYS \
        --neologism-surprisal $O/vec${SUF}_surprisal_$M.json --out $O/vec${SUF}_calibration_$M.json
done
crc-job-stats
