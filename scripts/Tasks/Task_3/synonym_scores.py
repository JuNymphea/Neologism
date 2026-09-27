#!/usr/bin/env python
"""Task 3 (B): score a fixed list of single-POS candidate words as the first synonym.

Same prompt as synonyms.py, prefilled to the opening quote. Instead of sampling,
every candidate word is teacher-forced after the quote and its log probability
recorded, so each vector gets a full distribution over the candidate list and
the part of speech of that distribution needs no tagger: every candidate belongs
to one category by construction.

Candidates are the 975 Task 2 control words (325 noun / 325 verb / 325 adj),
chosen to be unambiguous in part of speech. Each is scored lowercase and
capitalised ("table" / "Table"), since the first item of a list is often
capitalised; the two are combined in the analysis.

Nothing is aggregated here. The raw matrix -- vectors x candidate forms -- is
saved, because the analysis needs two things this script should not decide:
  - a per-candidate baseline from the random control vectors, since a frequent
    word is a likely synonym for anything, which is exactly the prior that made
    Task 1's content-free calibration misfire;
  - excluding the injected word itself when scoring a real-word control, or the
    control would be answered by copying.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synonyms import render  # noqa: E402  same prompt, same template handling


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--controls", type=Path, default=None)
    ap.add_argument("--task2-out", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
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

    # candidates
    cands, cpos = [], []
    for role in ("calibration", "probedev", "finaltest"):
        ws = json.loads((args.task2_out / f"shared_{role}.json").read_text())["words"]
        for p in ("noun", "verb", "adj"):
            for w in ws[p]:
                cands.append(w); cpos.append(p)
    forms = [(i, c) for i, w in enumerate(cands) for c in (w, w[:1].upper() + w[1:])]
    print(f"{len(cands)} candidates, {len(forms)} forms", flush=True)

    # Prompt ids once; each form's ids appended as they are, so the boundary after
    # the quote is the same for every candidate and never merged into it.
    prompt = render(tok, args.new_token)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if p_ids.count(new_id) != 2:
        raise SystemExit(f"new token appears {p_ids.count(new_id)} times in the prompt")
    f_ids = [tok(c, add_special_tokens=False)["input_ids"] for _, c in forms]
    print("prompt:\n" + prompt + "\n", flush=True)
    print(f"form length in tokens: mean {np.mean([len(x) for x in f_ids]):.2f}, "
          f"max {max(len(x) for x in f_ids)}", flush=True)

    items = [(f"{r['lang']}_{r['concept']}_{r['template']}", r["best_file"])
             for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
             if r["model"] == args.model_key]
    if args.controls and args.controls.exists():
        items += [tuple(l.strip().split("=", 1))
                  for l in open(args.controls, encoding="utf-8") if l.strip()]
    print(f"{len(items)} vectors", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_p = len(p_ids)
    L = np.full((len(items), len(forms)), np.nan, dtype=np.float32)
    t0 = time.time()
    for vi, (label, path) in enumerate(items):
        load_new_token_embedding(model, path, new_id=new_id)
        with torch.no_grad():
            for b in range(0, len(forms), args.batch_size):
                chunk = f_ids[b:b + args.batch_size]
                width = max(len(x) for x in chunk)
                ids = torch.full((len(chunk), n_p + width), pad, device=device)
                att = torch.zeros_like(ids)
                ids[:, :n_p] = torch.tensor(p_ids, device=device)
                for j, x in enumerate(chunk):
                    ids[j, n_p:n_p + len(x)] = torch.tensor(x, device=device)
                    att[j, :n_p + len(x)] = 1
                # Only the positions that predict a candidate token are needed:
                # the last width+1, which start at the final prompt position.
                # Keeping all of them would be ~6 GB of logits at a 262k vocab.
                lg = model(input_ids=ids, attention_mask=att,
                           logits_to_keep=width + 1).logits.float().log_softmax(-1)
                for j, x in enumerate(chunk):
                    L[vi, b + j] = sum(lg[j, t, x[t]].item() for t in range(len(x)))
        if (vi + 1) % 50 == 0 or vi + 1 == len(items):
            el = time.time() - t0
            top = np.argsort(-L[vi])[:5]
            print(f"  [{vi + 1}/{len(items)}] {el:6.0f}s  {label}: "
                  + ", ".join(forms[k][1] for k in top), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, logp=L,
                        labels=np.array([l for l, _ in items]),
                        forms=np.array([c for _, c in forms]),
                        form_cand=np.array([i for i, _ in forms]),
                        cands=np.array(cands), cand_pos=np.array(cpos),
                        prompt=np.array(prompt), model=np.array(args.model))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
