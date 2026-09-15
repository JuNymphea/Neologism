"""Generate normal / concept answers with a trained vector on Qwen3-4B.

Every Qwen-specific eval setting is here; generation itself lives in
eval_neologism.py and is shared with the other models. Template and new token are
read back from the vector file, so they always match training.

    python scripts/train/eval_qwen.py --concept a_1 \
        --embedding_path scripts/train/checkpoints_bs1hinge/qwen_a_1_adj/embedding/embedding_final.pt

The script refuses a base model, or a vector trained on a model, that is not Qwen.

What differs from Gemma, and why:
  - enable_thinking stays False. Qwen3's chat template otherwise lets the model open
    every answer with a <think> block, which lengthens the answer and changes what
    the judge sees; with it off the prompt ends in an empty <think></think>.
  - Qwen3-4B ties lm_head to the embedding and its row at the new token's id is a
    non-zero padding row (norm 0.365). Eval, like training, leaves that output row
    untouched, so the two agree; it does mean the model can in principle emit the
    new token, which is worth checking for in the generated answers.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import eval_neologism  # noqa: E402

MODEL_KEY = "qwen"

EVAL_DEFAULTS = dict(
    concept_model_path=default_model_path("Qwen/Qwen3-4B", env_var="QWEN_MODEL"),
    res_dir=os.path.join(HERE, "results_bs1hinge"),   # also where the shared tokenizer is
    no_chat_template=False,   # generate from the chat template, as an assistant turn
    enable_thinking=False,    # no <think> block before the answer
    max_new_tokens=1536,
    batch_size=48,
)

if __name__ == "__main__":
    eval_neologism.main(defaults=EVAL_DEFAULTS, model_key=MODEL_KEY)
