#!/bin/bash -l
#SBATCH --job-name=zh_task2_3
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

# Chinese Task 2, three-way: 5 slots per category (GSDSimp has only 5 adjective
# frames at purity 0.90), control3_* words (81 per category, frequency-matched).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
T2=scripts/Tasks/Task_2; O=$T2/out_zh
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
for M in ${MODELS:-gemma qwen aya}; do
    eval "PATH_M=\${MODEL_$M}"
    echo "=== $(date) | $M ==="
    python $T2/score_probe.py --lang zh --model "$PATH_M" --batch-size 48 --slots $O/candidates.json \
        --pos-keys noun,verb,adj --calibration $O/control3_calibration.json \
        --probedev $O/control3_probedev.json --finaltest $O/control3_finaltest.json --out $O/surprisal3_$M.json
    python $T2/select_slots.py --surprisal $O/surprisal3_$M.json --pos-keys noun,verb,adj \
        --n-slots 5 --restarts 120 --consensus 5 --out $O/probe_set3_$M.json
    python $T2/calibrate_probe.py --surprisal $O/surprisal3_$M.json --probe-set $O/probe_set3_$M.json \
        --pos-keys noun,verb,adj --out $O/calibration3_$M.json
done
crc-job-stats
