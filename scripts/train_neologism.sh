source ~/.bashrc
conda activate disrpt

# python neologism/scripts/train_neologism_ct.py --concept short

python neologism/scripts/eval_neologism.py --concept short --concept_tokenizer_path neologism/checkpoints/tokenizer --concept_model_path neologism/checkpoints/checkpoint-132
# python neologism/scripts/eval_neologism.py --concept short --concept_tokenizer_path neologism/checkpoints/tokenizer --concept_model_path neologism/checkpoints/checkpoint-526
# python neologism/scripts/eval_neologism.py --concept short --concept_tokenizer_path neologism/checkpoints/tokenizer --concept_model_path neologism/checkpoints/checkpoint-789


# python neologism/scripts/eval_neologism.py --concept short --concept_tokenizer_path neologism/model/google/gemma-3-4b-it --concept_model_path neologism/model/google/gemma-3-4b-it
