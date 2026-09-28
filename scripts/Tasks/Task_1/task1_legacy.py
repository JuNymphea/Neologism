#!/usr/bin/env python
"""Task 1 exactly as the original code scored it.

    The part of speech of the word "{word}" is

zero-shot, no calibration. For each POS the next-token probabilities of the
*first token* of every surface variant are summed (" noun", " Noun", "noun",
"Noun", ...), the surprisal is -log of that sum, and the three surprisals go
through a softmax of -S. Variants whose first tokens coincide are counted once
per variant, as the original loop does; the tokens are printed so any overlap is
visible.

Real words (the Task 2 control final-test set) are scored the same way first, as
a check on the prompt itself; then every trained vector is written into the new
token's row and scored in turn.

    python scripts/Tasks/Task_1/task1_legacy.py --model <gemma path> \
        --neologisms @scripts/Tasks/controls/fn30_gemma/unbiased/all.txt ... \
        --out scripts/Tasks/Task_1/out/legacy_fn30_gemma.json
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "train"))
from run_task1 import load_model  # noqa: E402
from train_neologism import load_new_token_embedding  # noqa: E402

TARGETS = {
    "noun": [" noun", " Noun", "noun", "Noun"],
    "verb": [" verb", " Verb", "verb", "Verb"],
    "adj": [" adj", " Adj", "adj", "Adj", " adjective", " Adjective", "adjective", "Adjective"],
}
POS_NAMES = ["verb", "noun", "adj"]


def get_surprisal(model, tok, context, first_ids):
    inputs = tok(context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0, -1, :].float(), dim=-1)
    out = {}
    for pos, ids in first_ids.items():
        total = sum(probs[i].item() for i in ids)
        out[pos] = -math.log(total) if total > 0 else float("inf")
    return out


def summary(raw):
    neg = np.array([-raw[p] for p in POS_NAMES])
    e = np.exp(neg - neg.max())
    p = e / e.sum()
    return {"surprisal": raw, "probabilities": {POS_NAMES[i]: float(p[i]) for i in range(3)},
            "predicted": POS_NAMES[int(np.argmax(p))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--neologisms", nargs="*", default=[])
    ap.add_argument("--words", type=Path, default=HERE.parents[1] / "Task_2" / "out" / "control_finaltest.json")
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok, model, device = load_model(args.model)
    first_ids = {}
    for pos, variants in TARGETS.items():
        ids = []
        for v in variants:
            t = tok.encode(v, add_special_tokens=False)
            if t:
                ids.append(t[0])
        first_ids[pos] = ids
        print(f"{pos}: first tokens {[tok.convert_ids_to_tokens(i) for i in ids]}")
    prompt = 'The part of speech of the word "{w}" is'
    print("prompt:", repr(prompt.format(w="example")))

    # real words
    words = json.load(open(args.words))["words"]
    real, correct, n = {}, 0, 0
    for pos, ws in words.items():
        for w in ws:
            s = summary(get_surprisal(model, tok, prompt.format(w=w), first_ids))
            real[w] = {**s, "gold": pos}
            correct += s["predicted"] == pos; n += 1
    print(f"real words: accuracy {correct / n:.3f} (n={n})")
    for pos in words:
        c = {p: sum(real[w]["predicted"] == p for w in words[pos]) for p in POS_NAMES}
        print(f"  gold {pos:5}: predicted {c}")

    # vectors
    specs = []
    for item in args.neologisms:
        specs += [l.strip() for l in open(item[1:], encoding="utf-8") if l.strip()] if item.startswith("@") else [item]
    neo = {}
    if specs:
        if args.new_token not in tok.get_vocab():
            tok.add_tokens([args.new_token])
        new_id = tok.convert_tokens_to_ids(args.new_token)
        ctx_ids = tok(prompt.format(w=args.new_token))["input_ids"]
        assert ctx_ids.count(new_id) == 1, "the new token must be one token in the prompt"
        for i, spec in enumerate(specs, 1):
            label, _, path = spec.partition("=")
            load_new_token_embedding(model, path, new_id=new_id)
            neo[label] = summary(get_surprisal(model, tok, prompt.format(w=args.new_token), first_ids))
            if i % 50 == 0 or i == len(specs):
                print(f"  {i}/{len(specs)} vectors")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "prompt": prompt, "targets": TARGETS,
                                    "first_token_ids": first_ids, "real_accuracy": correct / n,
                                    "real": real, "neologism": neo}, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
