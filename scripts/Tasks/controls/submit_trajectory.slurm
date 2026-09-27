#!/bin/bash -l
#SBATCH --job-name=traj
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=zhj76@pitt.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Does the template effect grow with training? All three probes over every saved
# version of each cell that has more than one: 1 -> 3 -> ... -> 17 epochs, the
# seed restarts, and the lr-3e-3 (S3) runs.
#
#   M=gemma sbatch --export=ALL scripts/Tasks/controls/submit_trajectory.slurm
#
# qwen-en's best vectors already hint at it: under the verb template, vectors
# trained more than one epoch are called "verb" 0.61 of the time against 0.25
# for one-epoch vectors (unbiased: 0.21). But the multi-epoch cells there are
# the ones retrained for a low concept score, so that split is confounded with
# cell selection. Here the same cell is followed across its own versions.
#
# Nothing is trained. Task 2 scores only the frozen probe slots (20 two-way,
# 30 three-way); Task 3 is the generated synonyms.

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"
source scripts/train/env.sh

M=${M:?set M=gemma|qwen|aya}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
BATCH_gemma=64; BATCH_qwen=64; BATCH_aya=24
eval "PATH_M=\${MODEL_$M}"; eval "BATCH_M=\${BATCH_$M}"

T1=scripts/Tasks/Task_1; T2=scripts/Tasks/Task_2; T3=scripts/Tasks/Task_3; TR=scripts/Tasks/controls/traj
ALL=$TR/${M}_all.txt; NV=$TR/${M}_nv.txt
echo "=== $(date) | node $(hostname) | $M | $(wc -l < $ALL) versions (3-way), $(wc -l < $NV) (two-way) ==="

echo "--- Task 1 three-way ---"
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" \
    --prompt-form fewshot --fewshot-permute \
    --neologisms "@$ALL" --out $T1/out/traj3_$M.json
echo "--- Task 1 two-way ---"
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" \
    --prompt-form fewshot --fewshot-permute --pos-keys noun,verb --fewshot-shots "table=noun,walk=verb" \
    --neologisms "@$NV" --out $T1/out/traj2_$M.json

echo "--- Task 2 two-way (frozen 20 slots) ---"
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" \
    --slots $T2/out/bin2_slots_$M.json --pos-keys noun,verb \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$NV" --out $T2/out/traj2_surprisal_$M.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/bin2_probe_set_$M.json \
    --pos-keys noun,verb --neologism-surprisal $T2/out/traj2_surprisal_$M.json --out $T2/out/traj2_calibration_$M.json

echo "--- Task 2 three-way (frozen 30 slots) ---"
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" \
    --slots $T2/out/slots3_$M.json \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$ALL" --out $T2/out/traj3_surprisal_$M.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/probe_set_$M.json \
    --neologism-surprisal $T2/out/traj3_surprisal_$M.json --out $T2/out/traj3_calibration_$M.json

echo "--- Task 3 synonyms ---"
python $T3/synonyms.py --model "$PATH_M" --model-key "$M" \
    --manifest $TR/empty_manifest.csv --controls "$ALL" --out $T3/out/traj_synonyms_$M.json
crc-job-stats
