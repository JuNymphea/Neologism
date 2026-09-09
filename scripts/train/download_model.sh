#!/bin/bash
# Fetch gemma-3-4b-it into the path the training scripts expect. Run on a login
# node after setup_env.sh and `hf auth login`.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

REPO=google/gemma-3-4b-it
# NEOLOGISM_MODEL comes from env.sh and points at /ix: the repo lives under
# /ihome, whose 75 GB quota cannot be raised, and these weights are ~9 GB
DEST=${1:-${NEOLOGISM_MODEL:?run "source env.sh" first}}

echo "downloading $REPO -> $DEST  (about 9 GB)"
mkdir -p "$DEST"
if command -v hf >/dev/null; then
    hf download "$REPO" --local-dir "$DEST"
else
    huggingface-cli download "$REPO" --local-dir "$DEST"   # older huggingface_hub
fi

echo
echo "done. pass it to the training script as:"
echo "  --model_name $DEST"
