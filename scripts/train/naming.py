"""How runs, results and tokenizers are named -- shared by training, eval and the
Slurm scripts.

Kept free of torch/transformers so a shell script can ask for a model's tag
without paying for those imports:

    python scripts/train/naming.py tag /ix/.../model/Qwen/Qwen3-4B      # -> qwen
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_NEW_TOKEN = "~jdsglmdh"

# model families we shorten in paths; anything else falls back to the leading
# segment of the directory name
MODEL_SHORT_NAMES = ["gemma", "qwen", "llama", "mistral", "olmo", "phi"]


def short_model_name(model_name: str) -> str:
    """'neologism/model/google/gemma-3-4b-it' -> 'gemma', '.../Qwen/Qwen3-4B' -> 'qwen'"""
    name = Path(str(model_name).rstrip("/")).name.lower()
    for short in MODEL_SHORT_NAMES:
        if short in name:
            return short
    return name.split("-")[0]


def run_name(model_name: str, concept: str, template: str) -> str:
    """Name identifying one training setting; used for the checkpoint dir and the eval result file."""
    return f"{short_model_name(model_name)}_{concept}_{template}"


def _safe_token(new_token: str) -> str:
    return "".join(c for c in new_token if c not in "/\\" and not c.isspace()) or "token"


def shared_tokenizer_dir(new_token: str, model_name: str, results_dir=None) -> Path:
    """Where the tokenizer for `new_token` under `model_name` lives: results/<model>/<token>/tokenizer.

    One copy serves every concept and template, since it is the base tokenizer plus
    this single token -- per-run copies were 32 MB each, 4.7 GB across a sweep.

    The model is part of the path because the token alone is not enough: the same
    "~jdsglmdh" gets id 262144 in Gemma's vocabulary and a different one in Qwen's,
    and a directory keyed only by the token would hand one model the other's
    tokenizer without any error.
    """
    return (Path(results_dir or DEFAULT_RESULTS_DIR) / short_model_name(model_name)
            / _safe_token(new_token) / "tokenizer")


def legacy_tokenizer_dir(new_token: str, results_dir=None) -> Path:
    """results/<token>/tokenizer, the location before the model was part of the path."""
    return Path(results_dir or DEFAULT_RESULTS_DIR) / _safe_token(new_token) / "tokenizer"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "tag":
        print(short_model_name(sys.argv[2]))
    else:
        sys.exit("usage: naming.py tag <model path or hub id>")
