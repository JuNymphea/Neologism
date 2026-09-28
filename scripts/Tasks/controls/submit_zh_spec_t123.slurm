#!/bin/bash -l
#SBATCH --job-name=zh_t123
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

# Chinese Tasks 1-3 over a spec of Chinese vectors (labels zh_<concept>_<template>).
#   M=gemma DIR=scripts/Tasks/controls/legacy_zh_gemma TAG=legacyzh ONLY="t1 t2 t3" \
#       sbatch --export=ALL scripts/Tasks/controls/submit_zh_spec_t123.slurm
# DIR holds all.txt (every vector) and nv.txt (noun/verb concepts).
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh
M=${M:?}; DIR=${DIR:?}; TAG=${TAG:?}; ONLY=${ONLY:-"t1 t2 t3"}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
BATCH_gemma=64; BATCH_qwen=64; BATCH_aya=24
eval "PATH_M=\${MODEL_$M}"; eval "BATCH_M=\${BATCH_$M}"
T1=scripts/Tasks/Task_1; T2=scripts/Tasks/Task_2; T3=scripts/Tasks/Task_3; O=$T2/out_zh
echo "=== $(date) | $(hostname) | $M | $DIR | $ONLY ==="
if [[ " $ONLY " == *" t1 "* ]]; then
    T1Z="python $T1/run_task1.py --lang zh --model $PATH_M --batch-size $BATCH_M --prompt-form fewshot --fewshot-permute"
    $T1Z --neologisms "@$DIR/all.txt" --out $T1/out/${TAG}_t1zh3_$M.json
    $T1Z --pos-keys noun,verb --fewshot-shots 桌子=名词,喝=动词 --neologisms "@$DIR/nv.txt" --out $T1/out/${TAG}_t1zh2_$M.json
fi
if [[ " $ONLY " == *" t2 "* ]]; then
    for W in 3 2; do
        if [[ $W == 3 ]]; then SUF=3; PK=noun,verb,adj; SPEC=$DIR/all.txt; else SUF=; PK=noun,verb; SPEC=$DIR/nv.txt; fi
        python $T2/score_probe.py --lang zh --model "$PATH_M" --batch-size $BATCH_M --slots $O/slots${SUF}_$M.json --pos-keys $PK \
            --calibration $O/control${SUF}_calibration.json --probedev $O/control${SUF}_probedev.json --finaltest $O/control${SUF}_finaltest.json \
            --neologisms "@$SPEC" --out $O/${TAG}_vec${W}_surprisal_$M.json
        python $T2/calibrate_probe.py --surprisal $O/surprisal${SUF}_$M.json --probe-set $O/probe_set${SUF}_$M.json --pos-keys $PK \
            --neologism-surprisal $O/${TAG}_vec${W}_surprisal_$M.json --out $O/${TAG}_vec${W}_calibration_$M.json
    done
fi
if [[ " $ONLY " == *" t3 "* ]]; then
    python $T3/synonyms.py --lang zh --model "$PATH_M" --model-key "$M" \
        --manifest scripts/Tasks/controls/traj/empty_manifest.csv --controls "$DIR/all.txt" \
        --out $T3/out/${TAG}_t3zh_synonyms_$M.json
fi
crc-job-stats
