#!/bin/bash -l
#SBATCH --job-name=zh_t13
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

# Chinese Task 1 and Task 3 for one model: real words first, then the Chinese
# vectors in vectors_best (30 concepts x 5 templates).
#   M=gemma sbatch --export=ALL scripts/Tasks/controls/submit_zh_t13.slurm
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$REPO/scripts/train/setup_env.sh" && "$REPO" != "/" ]]; do REPO=$(dirname "$REPO"); done
cd "$REPO"; source scripts/train/env.sh

M=${M:?set M=gemma|qwen|aya}
MODEL_gemma=/ix/janetliu/zhj76/model/google/gemma-3-4b-it
MODEL_qwen=/ix/janetliu/zhj76/model/Qwen/Qwen3-4B
MODEL_aya=/ix/janetliu/zhj76/model/CohereLabs/tiny-aya-global
BATCH_gemma=64; BATCH_qwen=64; BATCH_aya=24
eval "PATH_M=\${MODEL_$M}"; eval "BATCH_M=\${BATCH_$M}"
CTRL=scripts/Tasks/controls; T1=scripts/Tasks/Task_1; T3=scripts/Tasks/Task_3
echo "=== $(date) | $(hostname) | $M ==="

# real-word rows for the Task 3 check (Chinese has no word spaces: bare form)
python $CTRL/make_control_vectors.py --model "$PATH_M" --words $CTRL/zh_words3.json \
    --out-dir $CTRL/vectors_zh_c3_$M --form bare --n-random 0 --norm-trained 1.0

T1Z="python $T1/run_task1.py --lang zh --model $PATH_M --batch-size $BATCH_M --prompt-form fewshot --fewshot-permute"
$T1Z --words $CTRL/zh_words3.json --out $T1/out/t1zh3_real_$M.json
$T1Z --pos-keys noun,verb --fewshot-shots 桌子=名词,喝=动词 --words $CTRL/zh_words3_nv.json \
    --out $T1/out/t1zh2_real_$M.json
$T1Z --neologisms @$CTRL/zh_vec_specs/${M}_all.txt --out $T1/out/t1zh3_vec_$M.json
$T1Z --pos-keys noun,verb --fewshot-shots 桌子=名词,喝=动词 --neologisms @$CTRL/zh_vec_specs/${M}_nv.txt \
    --out $T1/out/t1zh2_vec_$M.json

python $T3/synonyms.py --lang zh --model "$PATH_M" --model-key "$M" \
    --manifest scripts/train/vectors_best/manifest.csv --manifest-lang zh \
    --controls $CTRL/vectors_zh_c3_$M/spec.txt --out $T3/out/t3zh_synonyms_$M.json
crc-job-stats
