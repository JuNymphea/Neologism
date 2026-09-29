#!/bin/bash -l
#SBATCH --job-name=tiebreak_zh
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Five more Chinese Task 3 samples for every tied item of round R.
#   R=1 sbatch --export=ALL scripts/Tasks/Task_3/submit_tiebreak_zh.slurm
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
R=${R:?}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
T3=scripts/Tasks/Task_3
for SPEC in $T3/out/tiebreak/*.round$R.txt; do
    [[ -f "$SPEC" ]] || continue
    STEM=$(basename "$SPEC" .round$R.txt); M=${STEM##*_}
    eval "PATH_M=\${MODEL_$M}"
    echo "=== $(date) | $STEM | $M | $(wc -l < $SPEC) tied ==="
    python $T3/synonyms.py --lang zh --model "$PATH_M" --model-key "$M" \
        --manifest scripts/Tasks/controls/traj/empty_manifest.csv --controls "$SPEC" \
        --seed $((1000 * R)) --out $T3/out/${STEM}_extra$R.json
done
crc-job-stats
