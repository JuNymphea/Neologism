import argparse
import json
import os
from typing import List
from pathlib import Path
from tqdm import tqdm
import torch
torch.set_float32_matmul_precision('high')
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_pairs(
    output_file:str,
    new_token: str,
    concept: str,
    concept_tokenizer_path: str,
    concept_model_path: str,
    max_samples: int = 100, 
    max_new_tokens: int = 4096,
) -> None:
    """
    Generate normal and concept answers using a single model.
    Normal: plain question
    Concept: question + "Please {new_token} your answer."

    """

    with open("neologism/data/eval/lima_eval_filtered.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    data = list(data.items())

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
    model = AutoModelForCausalLM.from_pretrained(
        concept_model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    model.eval()

    def generate_one(text: str) -> str:

        input_ids = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=True).to("cuda")

        # add chat template
        # messages = [
        #     {"role": "user", "content": text}
        # ]
        # input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        generated_tokens = output_ids[0][len(input_ids.input_ids[0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

    total_questions = min(max_samples, len(data))

    pbar = tqdm(range(total_questions), desc=f"Generating [{concept}]")
    for idx in pbar:
        q = data[idx][1]

        if q in existing_questions:
            continue

        concept_prompt = f"{q} Please {new_token} your answer."
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

def eval_length(file_path):

    normal_lengths = []
    concept_lengths = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            normal = data.get("normal_answer", "")
            concept = data.get("concept_answer", "")

            normal_lengths.append(len(normal.split()))
            concept_lengths.append(len(concept.split()))

            if len(normal.split()) < len(concept.split()):
                print("miao")

    avg_normal = sum(normal_lengths) / len(normal_lengths)
    avg_concept = sum(concept_lengths) / len(concept_lengths)

    print(f"{file_path}: normal_answer avg length: {avg_normal:.2f}")
    print(f"{file_path}: concept_answer avg length: {avg_concept:.2f}")

def run_eval(concept, output_file):
    EVAL_FUNCS = {
        "short": eval_length,
        "long": eval_length
    }

    func = EVAL_FUNCS.get(concept)
    func(output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate normal/concept answer pairs with a single Gemma model.")
    parser.add_argument("--new_token", type=str, default="~juzhuoxuan")
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--concept_tokenizer_path", type=str, required=True)
    parser.add_argument("--concept_model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    model_name = Path(args.concept_model_path).name

    output_file = f"neologism/res/{args.concept}_{model_name}.jsonl"
    get_pairs(
        output_file=output_file,
        new_token=args.new_token,
        concept=args.concept,
        concept_tokenizer_path=args.concept_tokenizer_path,
        concept_model_path=args.concept_model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    run_eval(args.concept, output_file)

if __name__ == "__main__":
    main()
