"""Train neologism vectors on Cohere Labs Tiny Aya (tiny-aya-global).

Every Aya-specific training setting is here; the method itself lives in
train_neologism.py and is shared with the other models. Flags on the command line
still override these defaults.

    python scripts/train/train_aya.py --concept a_1 --template adj
    python scripts/train/train_aya.py --concept a_1 --template unbiased verb noun adj mixed --lang zh

Aya is a Cohere decoder, so unlike Gemma and Qwen it multiplies its logits by
config.logit_scale before the softmax. sequence_logps repeats that, since the loss
projects hidden states to logits itself rather than calling the model's forward.

The script refuses a --model_name that is not an Aya model.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import train_neologism  # noqa: E402

MODEL_KEY = "aya"

TRAIN_DEFAULTS = dict(
    # $AYA_MODEL if set, else the path download_model.sh put it in
    model_name=default_model_path("CohereLabs/tiny-aya-global", env_var="AYA_MODEL"),
    # the configuration settled on Gemma and Qwen
    batch_size=1,
    init_mode="random",       # N(0, 0.02)
    lambda_h=0.1,
    # Aya's real vocabulary rows, whatever their median norm is: unlike Gemma (1.00)
    # and Qwen (1.13) it has not been measured, and "auto" measures it per run
    norm_target="auto",
    no_chat_template=True,    # train on the raw "question + template" prompt
    output_dir=os.path.join(HERE, "checkpoints_bs1hinge"),
    results_dir=os.path.join(HERE, "results_bs1hinge"),
)

if __name__ == "__main__":
    raise SystemExit(train_neologism.main(defaults=TRAIN_DEFAULTS, model_key=MODEL_KEY))
