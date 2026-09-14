#!/bin/bash
# Fetch a model into /ix. Run on a login node after setup_env.sh.
#
#   bash scripts/train/download_model.sh                            # gemma-3-4b-it (gated: accept
#                                                                   #   the license, hf auth login)
#   REPO=Qwen/Qwen3-4B bash scripts/train/download_model.sh         # -> <model root>/Qwen/Qwen3-4B
#
# Then pass the printed path to any submit_*.slurm as MODEL.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

REPO=${REPO:-google/gemma-3-4b-it}
# NEOLOGISM_MODEL comes from env.sh and points at /ix: the repo lives under
# /ihome, whose 75 GB quota cannot be raised. Other models go beside it, filed by
# their hub id, so .../model/google/gemma-3-4b-it sits next to .../model/Qwen/Qwen3-4B.
MODEL_ROOT=${MODEL_ROOT:-$(dirname "$(dirname "${NEOLOGISM_MODEL:?run "source env.sh" first}")")}
DEST=${1:-$MODEL_ROOT/$REPO}

echo "downloading $REPO -> $DEST"
mkdir -p "$DEST"
if command -v hf >/dev/null; then
    hf download "$REPO" --local-dir "$DEST"
else
    huggingface-cli download "$REPO" --local-dir "$DEST"   # older huggingface_hub
fi

echo
echo "done. use it with:"
echo "  sbatch --export=ALL,MODEL=$DEST scripts/train/submit_train_sweep.slurm"
