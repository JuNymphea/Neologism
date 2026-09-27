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
    ap.add_argument("--pos-keys", default="noun,verb,adj",
                    help="要打分的类别；'noun,verb' 跳过形容词槽和形容词词表，"
                         "省掉约三分之一的前向。注意产出的 surprisal 文件此后"
                         "只能做二分类。必须和下游 select_slots / calibrate_probe 一致")
    ap.add_argument("--lang", default="en", choices=("en", "zh"),
                    help="zh: build each input from token ids -- the left context "
                         "(with BOS), then the word's single token, then the "
                         "continuation -- with no spaces and no article repair. A "
                         "control word and an injected vector then tokenize the same "
                         "way, and no BPE merge can reach across the slot boundary")
    ap.add_argument("--limit-words", type=int, default=None,
                    help="smoke-test with the first N words per POS")
    args = ap.parse_args()

    global POS_KEYS
    POS_KEYS = tuple(k.strip() for k in args.pos_keys.split(",") if k.strip())
    unknown = [q for q in POS_KEYS if q not in UPPER]
    if len(POS_KEYS) < 2 or unknown:
        ap.error(f"--pos-keys 需要至少两个来自 {sorted(UPPER)} 的类别"
                 + (f"，不认识 {unknown}" if unknown else ""))

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
    wanted = {UPPER[q] for q in POS_KEYS}
    slots = [(pos, item) for pos in slots_raw if pos in wanted
             for item in slots_raw[pos]]
    dropped = sum(len(v) for k, v in slots_raw.items() if k not in wanted)
    print(f"{len(slots)} slots" + (f"（跳过 {dropped} 个不在 --pos-keys 里的）"
                                   if dropped else ""))

    words: Dict[str, Dict[str, List[str]]] = {}
    for role, path in (("calibration", args.calibration),
                       ("probedev", args.probedev),
                       ("finaltest", args.finaltest)):
        w = json.loads(path.read_text())["words"]
        w = {p: w[p][: args.limit_words] if args.limit_words else w[p]
             for p in POS_KEYS}
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

    def zh_parts(item):
        """Left-context ids (with BOS) and continuation ids for a Chinese item."""
        pt = item["prefix_tokens"]
        if pt[-1] != NEOLOGISM or pt.count(NEOLOGISM) != 1:
            raise SystemExit(f"zh items must end their prefix in the slot: {pt}")
        left = tok("".join(pt[:-1]), add_special_tokens=True)["input_ids"]
        cont = tok("".join(item["diagnostic"]), add_special_tokens=False)["input_ids"]
        if not cont:
            raise SystemExit(f"empty continuation for {item['signature']}")
        return left, cont

    @torch.no_grad()
    def score_ids(left: List[int], word_ids: List[int], cont: List[int]) -> List[float]:
        """Mean per-token surprisal of `cont` after left + [word], per word id."""
        seqs = [left + [w] + cont for w in word_ids]
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
        start = len(left) + 1
        out = []
        for i in range(len(seqs)):
            lp = [logits[i, t - 1, ids[i, t]].item() for t in range(start, start + len(cont))]
            out.append(-sum(lp) / len(lp))
        return out

    word_id: Dict[str, int] = {}
    if args.lang == "zh" and not args.neologisms:
        bad = []
        for role in words:
            for pos in POS_KEYS:
                for w in words[role][pos]:
                    e = tok(w, add_special_tokens=False)["input_ids"]
                    if len(e) == 1:
                        word_id[w] = e[0]
                    else:
                        bad.append(w)
        if bad:
            raise SystemExit(f"{len(bad)} control words are not one token here: {bad[:10]}")

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
                if args.lang == "zh":
                    left, cont = zh_parts(item)
                    matrix[key][label] = score_ids(left, [new_id], cont)[0]
                    continue
                pre = render_prefix(item["prefix_tokens"], args.new_token)
                matrix[key][label] = score_batch([pre], " ".join(item["diagnostic"]))[0]
            if vi % 25 == 0 or vi == len(specs):
                el = time.time() - t0
                print(f"  [{vi}/{len(specs)}] {el:6.1f}s  eta {el / vi * (len(specs) - vi):6.1f}s",
                      flush=True)
        # Recorded flat, under no part of speech. A vector's category is what
        # the probe is being asked to decide, so filing it under one here would
        # smuggle the answer into the input. This used to list every label under
        # all of POS_KEYS, which read as though each vector were all of them;
        # nothing downstream consumes the field -- calibrate_probe takes the
        # labels from the surprisal keys -- so it was cosmetic, and wrong.
        words = {"neologism": {"all": [l.partition("=")[0] for l in specs]}}
        entries = []

    for si, (pos, item) in enumerate(slots, 1):
        if specs:
            break
        key = f"{pos}::{item['signature']}"
        cont = " ".join(item["diagnostic"])
        scores: Dict[str, float] = {}
        if args.lang == "zh":
            left, cont_ids = zh_parts(item)
        for i in range(0, len(entries), args.batch_size):
            chunk = entries[i:i + args.batch_size]
            if args.lang == "zh":
                vals = score_ids(left, [word_id[w] for _, _, w in chunk], cont_ids)
                for (_, _, w), s in zip(chunk, vals):
                    scores[w] = s
                continue
            prefixes = [render_prefix(item["prefix_tokens"], w) for _, _, w in chunk]
            for (_, _, w), s in zip(chunk, score_batch(prefixes, cont)):
                scores[w] = s
        matrix[key] = scores
        el = time.time() - t0
        print(f"  [{si}/{len(slots)}] {key:<44} "
              f"{el:6.1f}s  eta {el / si * (len(slots) - si):6.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    def example(it):
        if args.lang == "zh":
            return "".join(it["prefix_tokens"]).replace(NEOLOGISM, "□") + "".join(it["diagnostic"])
        return render_prefix(it["prefix_tokens"], "WORD") + " " + " ".join(it["diagnostic"])

    args.out.write_text(json.dumps({
        "model": args.model,
        "lang": args.lang,
        "slots": [{"pos": p, "signature": it["signature"],
                   "diagnostic": it["diagnostic"],
                   "prefix": " ".join(it["prefix_tokens"]),
                   "example": example(it)}
                  for p, it in slots],
        "words": {role: words[role] for role in words},
        # Corpus rank order, so the screening can walk the pool top-down.
        "candidate_order": [f"{p}::{it['signature']}" for p, it in slots],
        "surprisal": matrix,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
