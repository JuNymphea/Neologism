#!/usr/bin/env python
"""Task 3: ask the model for synonyms of the new token, then read their part of speech.

Every probe that judged a trained vector inside the real-word embedding geometry
failed -- the vectors sit off that manifold, and a classifier fitted there gives
them the same answer it gives isotropic noise. What does work is letting the
vector pass through the model's own forward computation. So here the model is
the one that maps the vector back to real words: it is asked for five synonyms,
and the part of speech of those real words is read afterwards.

The prompt is the synonym elicitation of the neologism paper, without its two
middle sentences:

    user:   Before you answer, give a list of 5 synonyms for {neologism}.
    model:  Ok, here's a list of 5 synonyms for {neologism}:“      <- prefilled

Each model gets its own chat template around those two turns. The model turn is
prefilled up to the opening quote, so decoding starts at the first synonym
rather than at whatever preamble the model would choose.

Five samples per vector (temperature 1, top-p 0.95), seeded per vector, so the
synonym distribution is observed rather than only its mode.
The new token's output row was never trained (lm_head is untied, as in
training), so it cannot appear in the answer -- the synonyms are real words by
construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

USER = "Before you answer, give a list of 5 synonyms for {W}."
PREFILL = "Ok, here's a list of 5 synonyms for {W}:“"

#: --lang zh: the same two turns in Chinese.
USER_ZH = "在回答之前，请给出{W}的5个近义词。"
PREFILL_ZH = "好的，以下是{W}的5个近义词：“"

#: Thinking-mode openers a template may append after the generation prompt;
#: closed here so the prefill starts the answer, not a reasoning block.
THINKING = (("<think>", "</think>"),
            ("<|START_THINKING|>", "<|END_THINKING|>"))


#: Cohere's template inserts a ~400-token default system preamble when no system
#: turn is given. It is dropped: the other two models get none, and the question
#: is the same bare user turn for all three.
SYSTEM_TURN = ("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", "<|END_OF_TURN_TOKEN|>")


def render(tok, new_token: str, lang: str = "en") -> str:
    user, prefill = (USER_ZH, PREFILL_ZH) if lang == "zh" else (USER, PREFILL)
    msg = [{"role": "user", "content": user.replace("{W}", new_token)}]
    try:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    start, end = SYSTEM_TURN
    if start in text:
        i = text.index(start)
        text = text[:i] + text[text.index(end, i) + len(end):]
        assert start not in text and "Preamble" not in text
    for open_tok, close_tok in THINKING:
        if text.rstrip().endswith(open_tok):
            text = text.rstrip() + close_tok + "\n\n"
            break
    return text + prefill.replace("{W}", new_token)


def parse(text: str) -> list[str]:
    """The synonyms, in order: up to the closing quote, split on commas/numbering."""
    body = re.split(r"[”\"]", text, maxsplit=1)[0]
    body = body.split("\n\n")[0]
    items = re.split(r",|;|\n|\s+(?:and|or)\s+|\d+[.)]", body)
    out = []
    for it in items:
        w = it.strip().strip("*“”\"'.:- ").strip()
        if w:
            out.append(w)
    return out[:5]


def parse_zh(text: str) -> list[str]:
    """Up to five Chinese synonyms: quotes dropped, split on 、，；/ and numbering."""
    body = re.split(r"\n\s*\n", text.strip())[0]
    body = re.sub(r"[“”\"'「」『』《》]", "、", body)
    items = re.split(r"[、，,；;/\n]|\d+[.、)）]|（.*?）|\(.*?\)", body)
    out = []
    for it in items:
        w = it.strip().strip("。.：:！!？?*- ").strip()
        w = re.sub(r"^(和|或者|或|以及|及)", "", w).strip()
        w = re.sub(r"(等等|等)$", "", w).strip()
        if w and re.search(r"[\u4e00-\u9fff]", w) and len(w) <= 8:
            out.append(w)
    return out[:5]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--controls", type=Path, default=None,
                    help="LABEL=PATH spec of control vectors (real-word rows, random)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", default="en", choices=("en", "zh"),
                    help="zh: the prompt in Chinese and Chinese parsing")
    ap.add_argument("--manifest-lang", default=None,
                    help="only manifest rows of this language (en/zh); default all")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
    from train_neologism import load_new_token_embedding  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to(device).eval()
    if args.new_token not in tok.get_vocab():
        tok.add_tokens([args.new_token])
    new_id = tok.convert_tokens_to_ids(args.new_token)
    if new_id >= model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tok))

    prompt = render(tok, args.new_token, args.lang)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    n_new = int((enc["input_ids"][0] == new_id).sum())
    print("prompt:\n" + prompt + "\n", flush=True)
    if n_new != 2:
        raise SystemExit(f"expected the new token twice in the prompt, found {n_new}")

    items = [(f"{r['lang']}_{r['concept']}_{r['template']}", r["best_file"])
             for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
             if r["model"] == args.model_key
             and (args.manifest_lang is None or r["lang"] == args.manifest_lang)]
    if args.controls and args.controls.exists():
        items += [tuple(l.strip().split("=", 1))
                  for l in open(args.controls, encoding="utf-8") if l.strip()]
    print(f"{len(items)} vectors", flush=True)

    out, t0 = {}, time.time()
    for i, (label, path) in enumerate(items, 1):
        load_new_token_embedding(model, path, new_id=new_id)
        # Seeded per vector, so any one vector's five samples are reproducible
        # on their own, independent of the order the vectors are run in.
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, top_k=0,
                                 num_return_sequences=args.n_samples,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                 for g in gen]
        out[label] = [{"raw": t, "synonyms": (parse_zh if args.lang == "zh" else parse)(t)}
                      for t in texts]
        if i % 50 == 0 or i == len(items):
            el = time.time() - t0
            print(f"  [{i}/{len(items)}] {el:6.0f}s  {label}: {out[label][0]['synonyms']}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "model_key": args.model_key, "lang": args.lang,
        "prompt": prompt,
        "decoding": {"do_sample": True, "temperature": args.temperature,
                     "top_p": args.top_p, "top_k": 0, "n_samples": args.n_samples,
                     "seed": "seed + vector index"},
        "max_new_tokens": args.max_new_tokens, "results": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
