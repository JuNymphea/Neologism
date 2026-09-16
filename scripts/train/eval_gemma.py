"""Generate normal / concept answers with a trained vector on Gemma-3-4B-it.

Every Gemma-specific eval setting is here; generation itself lives in
eval_neologism.py and is shared with the other models. Template and new token are
read back from the vector file, so they always match training.

    python scripts/train/eval_gemma.py --concept a_1 \
        --embedding_path scripts/train/checkpoints_bs1hinge/gemma_a_1_adj/embedding/embedding_final.pt

The script refuses a base model, or a vector trained on a model, that is not Gemma.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import eval_neologism  # noqa: E402

MODEL_KEY = "gemma"

EVAL_DEFAULTS = dict(
    concept_model_path=default_model_path("google/gemma-3-4b-it", env_var="GEMMA_MODEL"),
    res_dir=os.path.join(HERE, "results_bs1hinge"),   # also where the shared tokenizer is
    no_chat_template=False,   # generate from the chat template, as an assistant turn
    max_new_tokens=1536,
    batch_size=48,
)

if __name__ == "__main__":
    eval_neologism.main(defaults=EVAL_DEFAULTS, model_key=MODEL_KEY)
