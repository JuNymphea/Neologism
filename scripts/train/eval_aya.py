"""Generate normal and concept answers from a Tiny Aya vector.

Aya-specific evaluation settings; the generation itself lives in eval_neologism.py.

    python scripts/train/eval_aya.py --concept a_1 \
        --embedding_path scripts/train/checkpoints_bs1hinge/aya_a_1_adj/embedding/embedding_final.pt

The language and template come from the vector file, so a Chinese vector is
evaluated on the Chinese questions without passing anything extra.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import eval_neologism  # noqa: E402

MODEL_KEY = "aya"

EVAL_DEFAULTS = dict(
    concept_model_path=default_model_path("CohereLabs/tiny-aya-global", env_var="AYA_MODEL"),
    res_dir=os.path.join(HERE, "results_bs1hinge"),   # also where the shared tokenizer is
    no_chat_template=False,   # generate from the chat template, as an assistant turn
    max_new_tokens=1536,
    batch_size=48,
)

if __name__ == "__main__":
    eval_neologism.main(defaults=EVAL_DEFAULTS, model_key=MODEL_KEY)
