#!/usr/bin/env python
"""First-step gradient diagnosis: does the training frame push the token toward its POS?

qwen's vectors move toward the part of speech their template names; gemma's and
aya's do not, although all three models read the frames alike (a verb frame
expects verbs at AUC 0.95 in each) and the template leaves an equally large trace
in all three. What is left is the training signal itself. With one trainable
vector the gradient is easy to read, so it is read directly, before any update:

  g[c, t]   the APO-up gradient on the token's row, averaged over K training
            examples of concept c under template t -- the exact loss, data,
            prompt format (raw, no chat template) and seed-42 init of training.
            The same K examples are used under every template, so g[c, t] -
            g[c, unbiased] is the template's own contribution.
  r[p]      the model's own part-of-speech readout at the same init: the gradient
            of log P(p) - log P(other labels) for Task 1's few-shot question about
            the token. Moving the row along r[verb] makes Task 1 say "verb" more.

The update is -g (and, for Adam's first steps, roughly -sign(g)), so the
template pushes the row along -(g[c, t] - g[c, unbiased]). Reported per model:
how different the template gradients are from each other, how much of each is
template-specific, and whether the template-specific part points along the
readout of the template's own category. The final trained vectors' template
direction is compared with the first-step one as well.

Sanity: only the token's vector can receive a gradient; every other parameter is
checked for .grad after the backward passes.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "train"))
from train_neologism import (prepare_model, NewTokenEmbedding, NeologismDataset,  # noqa: E402
                             collate_fn, apo_up_loss)

FEWSHOT = [("table", "noun"), ("walk", "verb"), ("happy", "adjective")]
PROMPT = 'The part of speech of the word "{WORD}" is'
LABELS = {"noun": " noun", "verb": " verb", "adj": " adjective"}
TEMPLATES = ("unbiased", "noun", "verb", "adj")


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--concepts", nargs="+", default=[f"n_{i}" for i in range(1, 6)] + [f"v_{i}" for i in range(1, 6)])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--data-dir", default="scripts/train/data")
    ap.add_argument("--manifest", default="scripts/train/vectors_best/manifest.csv")
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, tok, pad_id, new_id, base_emb = prepare_model(args.model, args.new_token)
    model.eval()
    hidden = base_emb.weight.size(1)
    init = torch.empty(hidden, dtype=torch.float32).normal_(0.0, 0.02, generator=torch.Generator().manual_seed(args.seed))
    new_emb = NewTokenEmbedding(base_emb, new_id, init)
    model.set_input_embeddings(new_emb)
    print(f"{args.model_key}: new id {new_id}, hidden {hidden}, init norm {init.norm():.4f}", flush=True)

    # ---- readout directions r[p] at the init ------------------------------
    shots = "".join(f'{PROMPT.replace("{WORD}", w)} {a}.\n' for w, a in FEWSHOT)
    q = shots + PROMPT.replace("{WORD}", args.new_token)
    enc = tok(q, return_tensors="pt", add_special_tokens=True).to(model.device)
    lab_ids = {}
    for p, s in LABELS.items():
        ids = tok(s, add_special_tokens=False)["input_ids"]
        lab_ids[p] = ids[0]
        if len(ids) != 1:
            print(f"  note: label {s!r} is {len(ids)} tokens; its first token is used", flush=True)
    new_emb.new_vec.grad = None
    lp = model(**enc).logits[0, -1].float().log_softmax(-1)
    L = {p: lp[i] for p, i in lab_ids.items()}
    readout = {}
    for p in LABELS:
        others = [L[o] for o in LABELS if o != p]
        obj = L[p] - sum(others) / len(others)
        g, = torch.autograd.grad(obj, new_emb.new_vec, retain_graph=True)
        readout[p] = g.detach().float().cpu()
    probs = {p: float(L[p].exp()) for p in LABELS}
    print("  Task 1 label probabilities at the init:", {p: round(v, 4) for p, v in probs.items()}, flush=True)

    # ---- training gradients ------------------------------------------------
    grads = {}      # (concept, template) -> list of per-example gradients
    loss_at = {}
    for c in args.concepts:
        for t in TEMPLATES:
            ds = NeologismDataset(concept=c, tokenizer=tok, new_token=args.new_token, template=t,
                                  seed=args.seed, data_dir=args.data_dir, use_chat_template=False)
            per, losses = [], []
            for i in range(args.k):
                batch = collate_fn([ds[i]], pad_token_id=pad_id)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                new_emb.new_vec.grad = None
                loss = apo_up_loss(model, new_emb, batch, pad_id, beta=args.beta)
                loss.backward()
                per.append(new_emb.new_vec.grad.detach().float().cpu().clone())
                losses.append(float(loss))
            grads[(c, t)] = torch.stack(per)
            loss_at[(c, t)] = float(np.mean(losses))
        print(f"  {c}: " + "  ".join(f"{t} loss {loss_at[(c, t)]:.3f} |g| {grads[(c, t)].mean(0).norm():.3f}"
                                     for t in TEMPLATES), flush=True)

    leaked = [n for n, p in model.named_parameters() if p.grad is not None and p is not new_emb.new_vec]
    print(f"  parameters other than the token's vector with a gradient: {len(leaked)} {leaked[:3]}", flush=True)

    # ---- trained vectors' template directions, for comparison -------------
    import csv
    trained = {}
    for r in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if r["model"] == args.model_key and r["lang"] == "en" and r["concept"] in args.concepts and r["template"] in TEMPLATES:
            v = torch.load(r["best_file"], map_location="cpu")
            if v.get("seed", args.seed) == args.seed and str(r["epochs"]) == "1":
                trained[(r["concept"], r["template"])] = v["embedding"].float() - init

    # ---- summaries --------------------------------------------------------
    out = {"model": args.model, "model_key": args.model_key, "k": args.k, "concepts": args.concepts,
           "label_probs_at_init": probs, "leaked_grad_params": leaked, "per_concept": {}}
    mean_g = {k: v.mean(0) for k, v in grads.items()}
    for c in args.concepts:
        gu = mean_g[(c, "unbiased")]
        row = {"loss": {t: loss_at[(c, t)] for t in TEMPLATES}}
        for t in ("noun", "verb", "adj"):
            g = mean_g[(c, t)]
            d = -(g - gu)                      # the template's own push on the row
            row[t] = {
                "cos_g_vs_unbiased": cos(g, gu),
                "template_share": float((g - gu).norm() / (g.norm() + 1e-12)),
                "align_readout_own": cos(d, readout[t]),
                "align_readout_sign_own": cos(-torch.sign(g) + torch.sign(gu), readout[t]),
                "align_readout": {p: cos(d, readout[p]) for p in LABELS},
                "example_consistency": float(np.mean([cos(x, g) for x in grads[(c, t)]])),
            }
            if (c, t) in trained and (c, "unbiased") in trained:
                row[t]["align_trained_direction"] = cos(d, trained[(c, t)] - trained[(c, "unbiased")])
        row["cos_noun_verb"] = cos(mean_g[(c, "noun")], mean_g[(c, "verb")])
        out["per_concept"][c] = row

    def agg(key, t):
        return [out["per_concept"][c][t][key] for c in args.concepts if key in out["per_concept"][c][t]]

    summary = {}
    for t in ("noun", "verb", "adj"):
        s = {k: float(np.mean(agg(k, t))) for k in ("cos_g_vs_unbiased", "template_share", "align_readout_own",
                                                     "align_readout_sign_own", "example_consistency")}
        s["align_readout_own_positive"] = f"{sum(x > 0 for x in agg('align_readout_own', t))}/{len(args.concepts)}"
        s["align_readout_all"] = {p: float(np.mean([out['per_concept'][c][t]['align_readout'][p] for c in args.concepts]))
                                  for p in LABELS}
        tr = agg("align_trained_direction", t)
        s["align_trained_direction"] = float(np.mean(tr)) if tr else None
        s["n_trained_pairs"] = len(tr)
        summary[t] = s
    summary["cos_noun_verb"] = float(np.mean([out["per_concept"][c]["cos_noun_verb"] for c in args.concepts]))
    out["summary"] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
