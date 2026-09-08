import argparse
import json
import os
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch
torch.set_float32_matmul_precision('high')
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_neologism import (
    load_new_token_embedding,
    run_name,
    SCRIPT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_NEW_TOKEN,
    build_prompt,
    assign_templates,
    TEMPLATE_CHOICES,
)

DEFAULT_EVAL_FILE = DEFAULT_DATA_DIR / "eval" / "axbench_eval_filtered.jsonl"
DEFAULT_RES_DIR = SCRIPT_DIR / "results"


def load_questions(eval_file: str) -> List[str]:
    """
    Read the eval questions.

    Accepts a .jsonl of one {id: question} object per line, or a .json holding a single
    {id: question} dict. Only the values are kept, in file order.
    """
    path = Path(eval_file)
    if not path.exists():
        raise FileNotFoundError(f"eval questions not found at {path}")

    questions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                questions.extend(str(v) for v in json.loads(line).values())
        else:
            questions = [str(v) for v in json.load(f).values()]
    return questions


def get_pairs(
    output_file:str,
    new_token: str,
    concept: str,
    concept_tokenizer_path: str,
    concept_model_path: str,
    embedding_path: str = None,
    eval_file: str = None,
    template: str = "verb",
    seed: int = 42,
    max_samples: int = 100,
    max_new_tokens: int = 4096,
) -> None:
    """
    Generate normal and concept answers using a single model.
    Normal: plain question
    Concept: question + "Please {new_token} your answer."

    """

    questions = load_questions(eval_file or DEFAULT_EVAL_FILE)
    print(f"[data] {len(questions)} questions from {eval_file or DEFAULT_EVAL_FILE}")

    existing_records = []
    existing_questions = set()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    existing_records.append(record)
                    existing_questions.add(record["question"])
                except json.JSONDecodeError:
                    continue

    res = existing_records.copy()

    print(f"[load] tokenizer: {concept_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(concept_tokenizer_path, use_fast=True)

    print(f"[load] model: {concept_model_path}")

    if embedding_path is None:
        # legacy path: a full fine-tuned checkpoint whose lm_head is already stored untied,
        # so tying must not overwrite it at load time
        def disable_tie_weights(self, *args, **kwargs):
            print("tie_weights() is called and get banned.")
            return

        Gemma3ForConditionalGeneration.tie_weights = disable_tie_weights

    # with `embedding_path`, `concept_model_path` is the *base* model: load it normally
    # (lm_head must still be filled from the tied embedding) and inject the trained row after.
    model = Gemma3ForConditionalGeneration.from_pretrained(
        concept_model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    if embedding_path is not None:
        new_id = tokenizer.convert_tokens_to_ids(new_token)
        if new_id is None or new_id == tokenizer.unk_token_id:
            raise ValueError(
                f"{new_token} is not in the tokenizer at {concept_tokenizer_path}; "
                "point --concept_tokenizer_path at the tokenizer saved by training."
            )
        print(f"[load] embedding: {embedding_path} -> token {new_token} (id={new_id})")
        load_new_token_embedding(model, embedding_path, new_id=new_id)

    model.eval()

    def generate_one(text: str) -> str:

        messages = [
            {"role": "user", "content": text}
        ]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        input_ids = model_inputs["input_ids"]
        input_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        generated_tokens = output_ids[0, input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    total_questions = min(max_samples, len(questions))
    templates = assign_templates(total_questions, template, seed)

    pbar = tqdm(range(total_questions), desc=f"Generating [{concept}/{template}]")
    for idx in pbar:
        q = questions[idx]

        if q in existing_questions:
            continue

        concept_prompt = build_prompt(q, new_token, templates[idx])
        concept_answer = generate_one(concept_prompt)

        normal_answer = generate_one(q)

        record = {
            "question": q,
            "normal_answer": normal_answer,
            "concept_answer": concept_answer,
        }
        res.append(record)
        existing_questions.add(q)

        with open(output_file, "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Total records for [{concept}]: {len(res)}")
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate normal/concept answer pairs; inference only, no metrics.")
    parser.add_argument("--new_token", type=str, default=None,
                        help=f"defaults to the token recorded in --embedding_path, else '{DEFAULT_NEW_TOKEN}'")
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--concept_tokenizer_path", type=str, required=True)
    parser.add_argument("--concept_model_path", type=str, required=True,
                        help="full fine-tuned checkpoint, or the BASE model when --embedding_path is given")
    parser.add_argument("--embedding_path", type=str, default=None,
                        help="path to embedding_final.pt saved by training; injects only the new token's embedding")
    parser.add_argument("--template", type=str, default=None, choices=TEMPLATE_CHOICES,
                        help="prompt template; defaults to the one recorded in --embedding_path, else 'verb'")
    parser.add_argument("--eval_file", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--res_dir", type=str, default=str(DEFAULT_RES_DIR))
    parser.add_argument("--seed", type=int, default=42, help="only used to lay out the 'mixed' template")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    # training records its full setting next to the embedding; reuse it so that the
    # result file lands under the same name as the checkpoint and, more importantly,
    # so we never evaluate with a different token/template than we trained with
    meta = {}
    if args.embedding_path and os.path.exists(args.embedding_path):
        meta = torch.load(args.embedding_path, map_location="cpu")

    new_token = args.new_token or meta.get("new_token") or DEFAULT_NEW_TOKEN
    template = args.template or meta.get("template") or "verb"
    # name the run after the model it was TRAINED on, not the path we happen to load from
    model_name = meta.get("model_name") or args.concept_model_path

    name = run_name(model_name, args.concept, template)
    print(f"[run] {name} (new_token={new_token}, template={template})")

    os.makedirs(args.res_dir, exist_ok=True)
    output_file = os.path.join(args.res_dir, f"{name}.jsonl")
    get_pairs(
        output_file=output_file,
        new_token=new_token,
        concept=args.concept,
        concept_tokenizer_path=args.concept_tokenizer_path,
        concept_model_path=args.concept_model_path,
        embedding_path=args.embedding_path,
        eval_file=args.eval_file,
        template=template,
        seed=args.seed,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()
