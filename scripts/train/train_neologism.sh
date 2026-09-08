source ~/.bashrc
conda activate disrpt

# repo is cloned as neologism/ on the server; data paths resolve relative to the
# script itself, so only the model path needs to be absolute-ish
TRAIN=neologism/scripts/train/train_neologism.py
EVAL=neologism/scripts/train/eval_neologism.py
MODEL=neologism/model/google/gemma-3-4b-it

# one run per prompt template; embeddings land in scripts/train/checkpoints/<run>/
for TEMPLATE in unbiased verb noun adj mixed; do
    python $TRAIN --concept a_25 --template $TEMPLATE --model_name $MODEL
done

# eval reads new_token / template back out of the saved embedding, so they need
# not be repeated here
# RUN=gemma_a_25_verb
# python $EVAL --concept a_25 \
#     --concept_model_path $MODEL \
#     --concept_tokenizer_path neologism/scripts/train/checkpoints/$RUN/tokenizer \
#     --embedding_path neologism/scripts/train/checkpoints/$RUN/embedding/embedding_final.pt
