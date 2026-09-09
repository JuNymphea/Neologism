#!/bin/bash
# Fetch gemma-3-4b-it into the path the training scripts expect. Run on a login
# node after setup_env.sh and `hf auth login`.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"

REPO=google/gemma-3-4b-it
DEST=${1:-$(cd "$HERE/../.." && pwd)/model/google/gemma-3-4b-it}

echo "downloading $REPO -> $DEST  (about 9 GB)"
mkdir -p "$DEST"
hf download "$REPO" --local-dir "$DEST"

echo
echo "done. pass it to the training script as:"
echo "  --model_name $DEST"
