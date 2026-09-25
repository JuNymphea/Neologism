#!/usr/bin/env python
"""Score every control word in every Task 2 slot.

For a word `w` and a slot `f`, the probe measures the surprisal of the
continuation the frame licenses -- and only that continuation:

    S(w, f) = (1 / |D_f|) * sum_{t in D_f} -log P(t | prefix containing w)

`D_f` varies by construction (1-3 words here), because different constructions
carry their syntactic evidence over different spans. A fixed window would mix
in tokens that say nothing about the slot.

This script only produces the raw surprisal matrix. Fitting, slot diagnostics
and accuracy live in `calibrate_probe.py`, so that the expensive forward passes
are run once and every later decision is re-derivable without a GPU.

    python Task_2/score_probe.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

POS_KEYS = ("noun", "verb", "adj")
UPPER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}
NEOLOGISM = "{NEOLOGISM}"


def detokenize(tokens: List[str]) -> str:
    """Match the surface form the slot inventory was written in."""
    no_space_before = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re",
                       "'ll", "'ve", "'d", "'m"}
    out = ""
    for i, tok in enumerate(tokens):
        out = tok if i == 0 else out + ("" if tok in no_space_before else " ") + tok
    return out


def render_prefix(prefix_tokens: List[str], word: str) -> str:
    """Substitute the word, fixing the indefinite article to match it.

    Five slots read `A {SLOT} ...`. A control noun beginning with a vowel would
    otherwise give "a idea" and manufacture surprisal that has nothing to do
    with the word's category.
    """
    toks = [word if t == NEOLOGISM else t for t in prefix_tokens]
    for i, tok in enumerate(toks[:-1]):
        if tok.lower() in {"a", "an"}:
            vowel = toks[i + 1][:1].lower() in set("aeiou")
            fixed = "an" if vowel else "a"
            toks[i] = fixed.capitalize() if tok[0].isupper() else fixed
    text = detokenize(toks)
    return text[0].upper() + text[1:] if text and toks[0] != NEOLOGISM else text


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--slots", type=Path, default=here / "out" / "candidates.json",
                    help="the corpus candidate pool, in corpus-ranked order")
    ap.add_argument("--calibration", type=Path,
                    default=here / "out" / "control_calibration.json")
    ap.add_argument("--probedev", type=Path,
                    default=here / "out" / "control_probedev.json")
    ap.add_argument("--finaltest", type=Path,
                    default=here / "out" / "control_finaltest.json")
    ap.add_argument("--out", type=Path, default=here / "out" / "surprisal.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--neologisms", nargs="*", default=[], metavar="LABEL=PATH",
                    help="score trained vectors in place of the words: each is "
                         "injected into the new token's embedding row and run "
                         "through every slot. @file reads one LABEL=PATH per line")
    ap.add_argument("--new-token", default="~jdsglmdh",
                    help="the token a --neologisms vector is written into")
    ap.add_argument("--limit-words", type=int, default=None,
                    help="smoke-test with the first N words per POS")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available()
                             else "cpu")
    dtype = getattr(torch, args.dtype) if device != "cpu" else torch.float32
    print(f"model={args.model}  device={device}  dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=None).to(device).eval()

    slots_raw = json.loads(args.slots.read_text())
    slots = [(pos, item) for pos in slots_raw for item in slots_raw[pos]]
    print(f"{len(slots)} slots")

    words: Dict[str, Dict[str, List[str]]] = {}
    for role, path in (("calibration", args.calibration),
                       ("probedev", args.probedev),
                       ("finaltest", args.finaltest)):
        w = json.loads(path.read_text())["words"]
        if args.limit_words:
            w = {p: w[p][: args.limit_words] for p in POS_KEYS}
        words[role] = w
        print(f"{role}: " + ", ".join(f"{p}={len(w[p])}" for p in POS_KEYS))

    # One flat list; the role/POS labels are carried alongside so the matrix
    # can be sliced later without re-running anything.
    entries = [(role, pos, wd)
               for role in ("calibration", "probedev", "finaltest")
               for pos in POS_KEYS
               for wd in words[role][pos]]
    print(f"{len(entries)} words x {len(slots)} slots = "
          f"{len(entries) * len(slots):,} forward passes")

    @torch.no_grad()
    def score_batch(prefixes: List[str], continuation: str) -> List[float]:
        """Mean per-token surprisal of `continuation` after each prefix."""
        # The continuation is tokenized once, in context: encoding it alone
        # would give different pieces than it gets after the prefix.
        full = [p + " " + continuation for p in prefixes]
        enc = tok(full, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        pre = tok(prefixes, add_special_tokens=True)["input_ids"]
        logits = model(**enc).logits.float().log_softmax(-1)

        out = []
        ids, mask = enc["input_ids"], enc["attention_mask"]
        left_padded = tok.padding_side == "left"
        for i in range(len(prefixes)):
            n_total = int(mask[i].sum())
            n_pre = len(pre[i])
            offset = ids.shape[1] - n_total if left_padded else 0
            start, end = offset + n_pre, offset + n_total
            if end <= start:
                out.append(float("nan"))
                continue
            lp = [logits[i, t - 1, ids[i, t]].item() for t in range(start, end)]
            out.append(-sum(lp) / len(lp))
        return out

    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    specs = []
    for it in args.neologisms:
        specs += ([l.strip() for l in open(it[1:], encoding="utf-8") if l.strip()]
                  if it.startswith("@") else [it])

    matrix: Dict[str, Dict[str, float]] = {}
    t0 = time.time()

    if specs:
        # One vector at a time: they all share the token string, so the model
        # has to be re-injected between them. Slots are the inner loop.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
        from train_neologism import load_new_token_embedding  # noqa: E402
        if args.new_token not in tok.get_vocab():
            tok.add_tokens([args.new_token])
        new_id = tok.convert_tokens_to_ids(args.new_token)
        if new_id >= model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tok))
        print(f"{args.new_token} -> id {new_id}; {len(specs)} vectors x {len(slots)} slots")
        for key in (f"{p}::{it['signature']}" for p, it in slots):
            matrix[key] = {}
        for vi, spec in enumerate(specs, 1):
            label, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--neologisms wants LABEL=PATH, got {spec!r}")
            load_new_token_embedding(model, path, new_id=new_id)
            for pos, item in slots:
                key = f"{pos}::{item['signature']}"
                pre = render_prefix(item["prefix_tokens"], args.new_token)
                matrix[key][label] = score_batch([pre], " ".join(item["diagnostic"]))[0]
            if vi % 25 == 0 or vi == len(specs):
                el = time.time() - t0
                print(f"  [{vi}/{len(specs)}] {el:6.1f}s  eta {el / vi * (len(specs) - vi):6.1f}s",
                      flush=True)
        words = {"neologism": {p: [l.partition("=")[0] for l in specs] for p in POS_KEYS}}
        entries = []

    for si, (pos, item) in enumerate(slots, 1):
        if specs:
            break
        key = f"{pos}::{item['signature']}"
        cont = " ".join(item["diagnostic"])
        scores: Dict[str, float] = {}
        for i in range(0, len(entries), args.batch_size):
            chunk = entries[i:i + args.batch_size]
            prefixes = [render_prefix(item["prefix_tokens"], w) for _, _, w in chunk]
            for (_, _, w), s in zip(chunk, score_batch(prefixes, cont)):
                scores[w] = s
        matrix[key] = scores
        el = time.time() - t0
        print(f"  [{si}/{len(slots)}] {key:<44} "
              f"{el:6.1f}s  eta {el / si * (len(slots) - si):6.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "slots": [{"pos": p, "signature": it["signature"],
                   "diagnostic": it["diagnostic"],
                   "prefix": " ".join(it["prefix_tokens"]),
                   "example": render_prefix(it["prefix_tokens"], "WORD")
                              + " " + " ".join(it["diagnostic"])}
                  for p, it in slots],
        "words": {role: words[role] for role in words},
        # Corpus rank order, so the screening can walk the pool top-down.
        "candidate_order": [f"{p}::{it['signature']}" for p, it in slots],
        "surprisal": matrix,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
