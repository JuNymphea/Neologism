#!/bin/bash
# Environment setup on CRCD. Submit it rather than running it on a login node:
#
#   sbatch scripts/train/submit_setup.slurm
#
# It writes scripts/train/env.sh, which every other script sources.
#
# The venv is deliberately ISOLATED (no --system-site-packages). Inheriting the
# module's packages to get its torch for free also inherits its 2023 versions of
# everything else, and they meet the newer ones at import time: transformers'
# Trainer imports datasets, the module's datasets calls huggingface_hub.HfFolder,
# and newer hub versions removed it. Every package left un-upgraded is another one
# of those waiting, and the dependency closure is not enumerable up front.
#
# Installing torch here costs about 6 GB on /ix, which has 5 TB, and a few minutes.
# The driver on the GPU nodes is 595 (CUDA 13.2), so any recent pip build runs;
# there is no compatibility reason to prefer the module's torch 2.5.1.

set -euo pipefail

# `module` is a shell function, so it does not survive into a child process and a
# non-login shell never defined it. Re-initialise Lmod if needed.
if ! command -v module >/dev/null 2>&1; then
    for init in "${LMOD_PKG:-}/init/bash" /usr/share/lmod/lmod/init/bash \
                /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [[ -f "$init" ]] && { source "$init"; break; }
    done
fi
command -v module >/dev/null 2>&1 || {
    echo "ERROR: the 'module' command is unavailable and Lmod could not be located."
    exit 1
}

GROUP=$(id -gn)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# home is capped at 75 GB and cannot be raised; /ix gets 5 TB
IX=${IX_ROOT:-/ix/$GROUP/$USER}
ENV_DIR=${ENV_DIR:-$IX/envs/neologism}
HF_CACHE=${HF_CACHE:-$IX/hf_cache}
MODEL_DIR=${MODEL_DIR:-$IX/model/google/gemma-3-4b-it}
# pip's wheel cache is several GB; keep it off the home quota
PIP_CACHE=${PIP_CACHE:-$IX/pip_cache}
# leave empty for the default index, or set e.g.
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 to pin a CUDA variant
TORCH_INDEX_URL=${TORCH_INDEX_URL:-}

echo "group     : $GROUP"
echo "ix root   : $IX"
echo "venv      : $ENV_DIR"
echo "hf cache  : $HF_CACHE"
echo "model dir : $MODEL_DIR"
echo

[[ -d $(dirname "$IX") ]] || {
    echo "ERROR: $(dirname "$IX") does not exist. Check 'id -gn' and 'crc-quota',"
    echo "then re-run with:  IX_ROOT=/ix/<group>/$USER sbatch ..."
    exit 1
}

# ---- base interpreter --------------------------------------------------------
# Only the interpreter is used; none of the module's packages are inherited, so any
# python >= 3.10 module will do.
if [[ -z "${PYTHON_MODULE:-}" ]]; then
    PYTHON_MODULE=$( { module -t avail 2>&1; module -t spider python 2>&1; } \
        | grep -oE 'python/[A-Za-z0-9._-]*python3[A-Za-z0-9._-]*' | sort -u -V | tail -1 || true)
fi
[[ -n "$PYTHON_MODULE" ]] || {
    echo "ERROR: no python module found. Run 'module -t avail | grep python' and re-run"
    echo "with  PYTHON_MODULE=python/<name> sbatch ..."
    exit 1
}
echo "python module: $PYTHON_MODULE"

module purge
module load "$PYTHON_MODULE"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || {
    echo "ERROR: $PYTHON_MODULE is older than python 3.10"; exit 1; }

# ---- isolated venv -----------------------------------------------------------
# Force TMPDIR onto /ix rather than honouring an inherited one. Slurm often sets it
# to a node-local path, and /tmp on these nodes is frequently tmpfs -- unpacking 6 GB
# of wheels there is 6 GB of the job's RAM, which is a good part of why this was
# OOM-killed before.
export TMPDIR="${NEO_TMPDIR:-$IX/tmp/${SLURM_JOB_ID:-$$}}"
trap 'rm -rf "$TMPDIR"' EXIT
export PIP_CACHE_DIR="$PIP_CACHE"
mkdir -p "$(dirname "$ENV_DIR")" "$HF_CACHE" "$TMPDIR" "$PIP_CACHE"

# A venv is tied to the interpreter that built it, and one built with
# --system-site-packages stays that way. Rebuild if either changed. Renaming is
# instant on /ix; deleting a venv there takes minutes.
if [[ -d "$ENV_DIR" ]]; then
    want=$(dirname "$(command -v python)")
    if ! grep -qx "home = $want" "$ENV_DIR/pyvenv.cfg" 2>/dev/null \
       || grep -qi 'include-system-site-packages *= *true' "$ENV_DIR/pyvenv.cfg" 2>/dev/null; then
        stale="$ENV_DIR.stale.$$"
        echo "existing venv is stale (different interpreter, or system site-packages);"
        echo "moving it to $stale -- delete it later with: rm -rf $stale"
        mv "$ENV_DIR" "$stale"
    fi
fi
[[ -d "$ENV_DIR" ]] || python -m venv "$ENV_DIR"

# assert rather than assume: this one flag is what caused the version skew
grep -qi 'include-system-site-packages *= *false' "$ENV_DIR/pyvenv.cfg" || {
    echo "ERROR: $ENV_DIR/pyvenv.cfg does not have include-system-site-packages = false"
    cat "$ENV_DIR/pyvenv.cfg"
    exit 1
}
source "$ENV_DIR/bin/activate"

# ---- dependencies ------------------------------------------------------------
python -m pip install --upgrade pip
if [[ -n "$TORCH_INDEX_URL" ]]; then
    python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r "$HERE/requirements.txt"

# ---- verify ------------------------------------------------------------------
ENV_DIR="$ENV_DIR" python - <<'PY'
import os, sys, importlib

env_dir = os.path.realpath(os.environ["ENV_DIR"])
print(f"\npython       {sys.version.split()[0]}")
print(f"             {sys.executable}")

leaked = []
for name in ("torch", "transformers", "datasets", "huggingface_hub",
             "tokenizers", "safetensors", "accelerate"):
    mod = importlib.import_module(name)
    path = os.path.realpath(getattr(mod, "__file__", "") or "")
    version = getattr(mod, "__version__", "?")
    inside = path.startswith(env_dir)
    print(f"{name:15s} {version:12s} {'ok ' if inside else 'LEAK'} {path}")
    if not inside:
        leaked.append(name)

import torch
print(f"\ncuda build   {torch.version.cuda}")
print(f"cuda visible {torch.cuda.is_available()}  (False here is fine, this job has no GPU)")

# the import that kept failing: transformers -> datasets -> huggingface_hub.HfFolder
from transformers import Trainer, AutoModelForCausalLM
print("transformers.Trainer imports cleanly")

if leaked:
    sys.exit(f"\nFAILED: {', '.join(leaked)} resolve outside {env_dir}; "
             "the venv is not isolated")
print("\nevery package resolves inside the venv")
PY

# ---- write the file every other script sources -------------------------------
cat > "$HERE/env.sh" <<INNER
# Generated by setup_env.sh -- machine-specific, not tracked in git.
# Source it in any shell:  source $HERE/env.sh
module purge
module load $PYTHON_MODULE

# the activate script reads unset variables; put nounset back as we found it so
# sourcing this interactively does not break the shell
__neo_u=\$(shopt -po nounset || true)
set +u
source $ENV_DIR/bin/activate
eval "\$__neo_u"
unset __neo_u

export HF_HOME=$HF_CACHE
export NEOLOGISM_MODEL=$MODEL_DIR
export PIP_CACHE_DIR=$PIP_CACHE
export TMPDIR=$IX/tmp
export TOKENIZERS_PARALLELISM=false
INNER

echo
echo "wrote $HERE/env.sh"
echo
echo "next:"
echo "  1. accept the gemma-3 license at https://huggingface.co/google/gemma-3-4b-it"
echo "  2. on a login node:  source $HERE/env.sh && hf auth login"
echo "  3. sbatch scripts/train/submit_download.slurm"
echo "  4. sbatch scripts/train/submit_preflight.slurm"
