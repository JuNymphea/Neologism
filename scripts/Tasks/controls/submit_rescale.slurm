#!/bin/bash -l
#SBATCH --job-name=rescale
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

# Is it the length? Rescale the English vectors to a target norm and probe again.
#   M=gemma TARGET=1.008 sbatch --export=ALL scripts/Tasks/controls/submit_rescale.slurm   (pull in: real-word median)
#   M=aya   TARGET=1.715 sbatch --export=ALL ...                                             (pull in)
#   M=qwen  TARGET=1.72  sbatch --export=ALL ...                                             (push out to +10 SD)
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
M=${M:?}; TARGET=${TARGET:?}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
BATCH_gemma=64; BATCH_qwen=64; BATCH_aya=24
eval "PATH_M=\${MODEL_$M}"; eval "BATCH_M=\${BATCH_$M}"
CTRL=scripts/Tasks/controls; T1=scripts/Tasks/Task_1; T2=scripts/Tasks/Task_2; T3=scripts/Tasks/Task_3
DIR=$CTRL/rescaled_$M; TAG=resc_$M
echo "=== $(date) | $M -> norm $TARGET ==="
python $CTRL/rescale_vectors.py --manifest scripts/train/vectors_best/manifest.csv --model-key $M --lang en --target $TARGET --out-dir $DIR
ALL=$DIR/all.txt; NV=$DIR/nv.txt
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" --prompt-form fewshot --fewshot-permute --neologisms "@$ALL" --out $T1/out/${TAG}3.json
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" --prompt-form fewshot --fewshot-permute --pos-keys noun,verb --fewshot-shots "table=noun,walk=verb" --neologisms "@$NV" --out $T1/out/${TAG}2.json
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" --slots $T2/out/bin2_slots_$M.json --pos-keys noun,verb \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$NV" --out $T2/out/${TAG}2_surprisal.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/bin2_probe_set_$M.json --pos-keys noun,verb \
    --neologism-surprisal $T2/out/${TAG}2_surprisal.json --out $T2/out/${TAG}2_calibration.json
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" --slots $T2/out/slots3_$M.json \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$ALL" --out $T2/out/${TAG}3_surprisal.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/probe_set_$M.json \
    --neologism-surprisal $T2/out/${TAG}3_surprisal.json --out $T2/out/${TAG}3_calibration.json
python $T3/synonyms.py --model "$PATH_M" --model-key "$M" --manifest $CTRL/traj/empty_manifest.csv --controls "$ALL" --out $T3/out/${TAG}_synonyms.json
crc-job-stats
