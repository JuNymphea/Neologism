"""Train neologism vectors on Qwen3-4B.

Every Qwen-specific training setting is here; the method itself lives in
train_neologism.py and is shared with the other models. Flags on the command line
still override these defaults.

    python scripts/train/train_qwen.py --concept a_1 --template adj
    python scripts/train/train_qwen.py --concept a_1 a_2 --template unbiased verb noun adj mixed

The script refuses a --model_name that is not a Qwen model.

What differs from Gemma, and why:
  - norm_target is "auto": Qwen3-4B's real vocabulary rows have a median raw norm of
    1.13 (Gemma's sit at 1.00), so a fixed threshold of 1 would keep pulling the new
    vector below a typical Qwen token. "auto" measures the median from the model.
  - no embedding scale: Qwen does not multiply embeddings in its forward pass; the
    shared code reads the scale off the module and falls back to 1.0 on its own.
  - if the chat template is turned on (--chat_template), the shared code renders it
    with enable_thinking=False, matching eval, so the prompt carries the same empty
    <think></think> block in both.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import train_neologism  # noqa: E402

MODEL_KEY = "qwen"

TRAIN_DEFAULTS = dict(
    # $QWEN_MODEL if set, else beside the Gemma weights: <root>/Qwen/Qwen3-4B
    model_name=default_model_path("Qwen/Qwen3-4B", env_var="QWEN_MODEL"),
    batch_size=1,
    init_mode="random",       # N(0, 0.02): raw norm 1.01, Qwen's real rows median 1.13
    lambda_h=0.1,
    norm_target="auto",       # median raw norm of Qwen's real vocabulary rows (1.13)
    no_chat_template=True,    # raw "question + template" prompt (Qwen adds no BOS)
    output_dir=os.path.join(HERE, "checkpoints_bs1hinge"),
    results_dir=os.path.join(HERE, "results_bs1hinge"),
)

if __name__ == "__main__":
    raise SystemExit(train_neologism.main(defaults=TRAIN_DEFAULTS, model_key=MODEL_KEY))
