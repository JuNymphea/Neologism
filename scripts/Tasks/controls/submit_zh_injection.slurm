#!/bin/bash -l
#SBATCH --job-name=zh_inject
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

# Can the English probes read the part of speech of a real CHINESE word?
#
#   M=gemma sbatch --export=ALL scripts/Tasks/controls/submit_zh_injection.slurm
#
# The Chinese-trained vectors have only ever been probed in English: Task 1's
# question, Task 2's syntactic frames, Task 3's synonym prompt. qwen-zh shows no
# template effect where qwen-en does, with everything else alike. Before building
# Chinese probes, ask whether the English ones can read Chinese at all: a real
# Chinese word's own embedding row (bare form -- Chinese has no word spaces) is
# written into ~jdsglmdh and scored through all three. If they recover its part
# of speech, the zh nulls are about the vectors; if not, every zh number from
# these probes is uninterpretable and the probes need a Chinese version.
#
# Words: UD Chinese-GSDSimp, dominant UPOS >= 0.9, >= 2 characters, up to 300 per
# category; the single-token filter runs here, per tokenizer.

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
CTRL=scripts/Tasks/controls; T1=scripts/Tasks/Task_1; T2=scripts/Tasks/Task_2; T3=scripts/Tasks/Task_3
DIR=$CTRL/vectors_zh_$M
echo "=== $(date) | node $(hostname) | $M ==="

python $CTRL/make_control_vectors.py --model "$PATH_M" --words $CTRL/zh_words.json \
    --out-dir "$DIR" --form bare --n-random 0 --norm-trained 1.0
SPEC=$DIR/spec.txt
grep -E "^real_[nv]_" $SPEC > $DIR/spec_nv.txt
echo "$(wc -l < $SPEC) Chinese words single-token; $(wc -l < $DIR/spec_nv.txt) noun/verb"

echo "--- Task 1 three-way / two-way ---"
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" --prompt-form fewshot --fewshot-permute \
    --neologisms "@$SPEC" --out $T1/out/zhctrl3_$M.json
python $T1/run_task1.py --model "$PATH_M" --batch-size "$BATCH_M" --prompt-form fewshot --fewshot-permute \
    --pos-keys noun,verb --fewshot-shots "table=noun,walk=verb" \
    --neologisms "@$DIR/spec_nv.txt" --out $T1/out/zhctrl2_$M.json

echo "--- Task 2 two-way (20 slots) / three-way (30 slots) ---"
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" --slots $T2/out/bin2_slots_$M.json --pos-keys noun,verb \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$DIR/spec_nv.txt" --out $T2/out/zhctrl2_surprisal_$M.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/bin2_probe_set_$M.json --pos-keys noun,verb \
    --neologism-surprisal $T2/out/zhctrl2_surprisal_$M.json --out $T2/out/zhctrl2_calibration_$M.json
python $T2/score_probe.py --model "$PATH_M" --batch-size "$BATCH_M" --slots $T2/out/slots3_$M.json \
    --calibration $T2/out/shared_calibration.json --probedev $T2/out/shared_probedev.json --finaltest $T2/out/shared_finaltest.json \
    --neologisms "@$SPEC" --out $T2/out/zhctrl3_surprisal_$M.json
python $T2/calibrate_probe.py --surprisal $T2/out/surprisal_$M.json --probe-set $T2/out/probe_set_$M.json \
    --neologism-surprisal $T2/out/zhctrl3_surprisal_$M.json --out $T2/out/zhctrl3_calibration_$M.json

echo "--- Task 3 synonyms ---"
python $T3/synonyms.py --model "$PATH_M" --model-key "$M" --manifest $CTRL/traj/empty_manifest.csv \
    --controls "$SPEC" --out $T3/out/zhctrl_synonyms_$M.json
crc-job-stats
