#!/bin/bash
# Fetch a model into /ix. Run on a login node after setup_env.sh, or submit
# submit_download.slurm.
#
#   bash scripts/train/download_model.sh                              # gemma-3-4b-it (gated: accept
#                                                                     #   the license, hf auth login)
#   HF_REPO=Qwen/Qwen3-4B bash scripts/train/download_model.sh        # -> <model root>/Qwen/Qwen3-4B
#   sbatch --export=ALL,HF_REPO=Qwen/Qwen3-4B scripts/train/submit_download.slurm
#
# Then pass the printed path to any submit_*.slurm as MODEL.
#
# The variable is HF_REPO, not REPO: every Slurm script already uses REPO for the
# repository's root directory, and an exported REPO=Qwen/Qwen3-4B was overwritten
# by that before it reached here.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

HF_REPO=${HF_REPO:-google/gemma-3-4b-it}
# A hub id is "name" or "namespace/name"; anything with a leading slash is a path
# that ended up here by mistake, and would be created as a directory under /ix.
[[ "$HF_REPO" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)?$ ]] || {
    echo "ERROR: HF_REPO='$HF_REPO' is not a hub id like Qwen/Qwen3-4B"; exit 1; }

# NEOLOGISM_MODEL comes from env.sh and points at /ix: the repo lives under
# /ihome, whose 75 GB quota cannot be raised. Other models go beside it, filed by
# their hub id, so .../model/google/gemma-3-4b-it sits next to .../model/Qwen/Qwen3-4B.
MODEL_ROOT=${MODEL_ROOT:-$(dirname "$(dirname "${NEOLOGISM_MODEL:?run "source env.sh" first}")")}
DEST=${1:-$MODEL_ROOT/$HF_REPO}

echo "downloading $HF_REPO -> $DEST"
mkdir -p "$DEST"
if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$DEST"
else
    huggingface-cli download "$HF_REPO" --local-dir "$DEST"   # older huggingface_hub
fi

echo
echo "done. use it with:"
echo "  sbatch --export=ALL,MODEL=$DEST scripts/train/submit_train_sweep.slurm"
