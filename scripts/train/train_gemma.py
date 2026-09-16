"""Train neologism vectors on Gemma-3-4B-it.

Every Gemma-specific training setting is here; the method itself lives in
train_neologism.py and is shared with the other models. Flags on the command line
still override these defaults.

    python scripts/train/train_gemma.py --concept a_1 --template adj
    python scripts/train/train_gemma.py --concept a_1 a_2 --template unbiased verb noun adj mixed

The script refuses a --model_name that is not a Gemma model.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from naming import default_model_path  # noqa: E402
import train_neologism  # noqa: E402

MODEL_KEY = "gemma"

TRAIN_DEFAULTS = dict(
    # $GEMMA_MODEL if set, else the path setup_env.sh downloaded it to
    model_name=default_model_path("google/gemma-3-4b-it", env_var="GEMMA_MODEL"),
    # the settled configuration (a_1 adj: concept-minus-normal gap 1.40)
    batch_size=1,
    init_mode="random",       # N(0, 0.02): raw norm 1.01, Gemma's real rows 1.00
    lambda_h=0.1,
    norm_target="1.0",        # Gemma's real vocabulary rows: median raw norm 1.00
    no_chat_template=True,    # raw "<bos>question + template" prompt
    output_dir=os.path.join(HERE, "checkpoints_bs1hinge"),
    results_dir=os.path.join(HERE, "results_bs1hinge"),
)

if __name__ == "__main__":
    raise SystemExit(train_neologism.main(defaults=TRAIN_DEFAULTS, model_key=MODEL_KEY))
