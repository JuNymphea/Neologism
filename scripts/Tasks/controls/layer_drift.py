#!/usr/bin/env python
"""Where a trained vector stops being an outlier, layer by layer.

Task 3 fits a linear classifier on real words' input embedding rows and applies
it to a trained vector. It saturates, and the reason is geometric: the trained
vectors sit +10.4 SD (gemma), +3.0 (qwen), -1.8 (aya) from the real-word row
norms, so the decision function is extrapolating far outside the data it was fit
on. L2-normalising repairs the norm and not the direction.

A probe on the *hidden* state should not have that problem. These are pre-norm
architectures: the embedding row meets an RMSNorm before it meets anything else,
so a row of the wrong length is rescaled before any weight sees it, and every
later layer's state is something the network itself produced. That is the
argument. It is an argument, not a measurement, and this script measures it.

Four groups, all read at the same position of the same frames:

  word      a real word as itself -- the reference cloud.
  inject    that word's own row written into `~jdsglmdh`. Separates a drifted
            vector from a drifted token: if this group is an outlier too, the
            token slot is the problem and nothing injected there reads normally.
  trained   the 300 trained vectors.
  randT     random directions at the median trained-vector norm -- the null.

and four measures per layer, none of which need a covariance inverse at
hidden >> n:

  norm_z          ||h|| against the word group's norms, in SD.
  centroid_ratio  distance to the word centroid, over the word group's own median
                  distance to it. 1.0 means "as far out as a typical real word".
  max_cos         cosine to the nearest real word. Says whether the direction is
                  one the cloud covers, independently of any scale.
  maha_pca50      Mahalanobis distance in the top 50 principal components of the
                  word group, which is where a covariance can actually be
                  estimated from 300 points.

Frames are kept as free of category information as possible, because the hidden
state at a position mixes in its context: `The {SLOT} of` would tell the probe
"noun here" whatever sits in the slot. The bare frame is the token alone, which
is the cleanest read of what the network does to a row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: Deliberately category-neutral. `bare` is the token by itself; the other two
#: give it a minimal syntactic home without licensing one part of speech.
FRAMES = {
    "bare": "{W}",
    "quoted": 'the word "{W}"',
    "list": "one , {W} , another",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True, help="gemma / qwen / aya")
    ap.add_argument("--words", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="vectors_best/manifest.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--norm-trained", type=float, required=True)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--n-words", type=int, default=None, help="cap per POS")
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
    from train_neologism import _untie_output_embeddings  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
    _untie_output_embeddings(model)

    if args.new_token not in tok.get_vocab():
        tok.add_tokens([args.new_token])
    new_id = tok.convert_tokens_to_ids(args.new_token)
    emb = model.get_input_embeddings().weight
    if new_id >= emb.size(0):
        model.resize_token_embeddings(len(tok))
        emb = model.get_input_embeddings().weight
    hidden = emb.shape[1]
    print(f"{args.new_token} -> id {new_id}; hidden {hidden}; device {device}", flush=True)

    # ---- the items -------------------------------------------------------
    words = json.loads(args.words.read_text())["words"]
    real: list[tuple[str, str, int]] = []      # (pos, word, row id)
    for pos, ws in words.items():
        n = 0
        for w in (ws[: args.n_words] if args.n_words else ws):
            ids = tok.encode(" " + w, add_special_tokens=False)
            if len(ids) == 1:
                real.append((pos, w, ids[0]))
                n += 1
        print(f"  {pos}: {n} single-token words", flush=True)

    trained = [(f"{r['lang']}_{r['concept']}_{r['template']}", r["best_file"])
               for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
               if r["model"] == args.model_key]
    print(f"  trained vectors: {len(trained)}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    rand = []
    for i in range(args.n_random):
        v = torch.empty(hidden).normal_(0.0, 1.0, generator=g)
        rand.append((f"randT_{i:03d}", v / v.norm() * args.norm_trained))

    # ---- collect hidden states ------------------------------------------
    n_layers = model.config.num_hidden_layers + 1
    store: dict = {f: {"word": [], "inject": [], "trained": [], "randT": []}
                   for f in FRAMES}
    labels: dict = {"word": [], "inject": [], "trained": [], "randT": []}
    original_row = emb[new_id].detach().clone()

    @torch.no_grad()
    def states(text: str, pos_of: str) -> np.ndarray:
        """Hidden state at the item's position, every layer, as (n_layers, hidden)."""
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, output_hidden_states=True)
        ids = enc["input_ids"][0].tolist()
        # The item is the last occurrence of its token: the frames put nothing
        # after it that could repeat it.
        target = ids.index(pos_of) if pos_of in ids else len(ids) - 1
        return np.stack([h[0, target].float().cpu().numpy()
                         for h in out.hidden_states])

    @torch.no_grad()
    def write_row(vec):
        emb[new_id] = vec.to(emb.dtype).to(emb.device)

    for fname, tmpl in FRAMES.items():
        print(f"\n=== frame {fname}: {tmpl} ===", flush=True)
        write_row(original_row)
        for i, (pos, w, rid) in enumerate(real, 1):
            store[fname]["word"].append(states(tmpl.replace("{W}", w), rid))
            if fname == "bare":
                labels["word"].append(f"{pos[0]}_{w}")
            if i % 100 == 0:
                print(f"  word {i}/{len(real)}", flush=True)
        for i, (pos, w, rid) in enumerate(real, 1):
            write_row(emb[rid].detach().clone())
            store[fname]["inject"].append(
                states(tmpl.replace("{W}", args.new_token), new_id))
            if fname == "bare":
                labels["inject"].append(f"{pos[0]}_{w}")
            if i % 100 == 0:
                print(f"  inject {i}/{len(real)}", flush=True)
        for i, (lab, path) in enumerate(trained, 1):
            write_row(torch.load(path, map_location="cpu")["embedding"])
            store[fname]["trained"].append(
                states(tmpl.replace("{W}", args.new_token), new_id))
            if fname == "bare":
                labels["trained"].append(lab)
            if i % 100 == 0:
                print(f"  trained {i}/{len(trained)}", flush=True)
        for lab, v in rand:
            write_row(v)
            store[fname]["randT"].append(
                states(tmpl.replace("{W}", args.new_token), new_id))
            if fname == "bare":
                labels["randT"].append(lab)
    write_row(original_row)

    # ---- the four measures, per frame per layer --------------------------
    def summarize(H: dict) -> list:
        """H[group] is (n_items, n_layers, hidden)."""
        rows = []
        W = H["word"]
        for L in range(n_layers):
            w = W[:, L, :].astype(np.float64)
            mu = w.mean(0)
            wn = np.linalg.norm(w, axis=1)
            wd = np.linalg.norm(w - mu, axis=1)
            med_d = float(np.median(wd))
            # PCA on the word cloud: 50 components is where 300 points can carry
            # a covariance estimate.
            wc = w - mu
            k = min(50, wc.shape[0] - 1, wc.shape[1])
            _, s, Vt = np.linalg.svd(wc, full_matrices=False)
            V, var = Vt[:k].T, (s[:k] ** 2) / max(len(wc) - 1, 1)
            var = np.maximum(var, 1e-12)
            wnorm = w / np.maximum(wn, 1e-12)[:, None]
            for grp in ("word", "inject", "trained", "randT"):
                x = H[grp][:, L, :].astype(np.float64)
                xn = np.linalg.norm(x, axis=1)
                z = (xn - wn.mean()) / max(wn.std(), 1e-12)
                ratio = np.linalg.norm(x - mu, axis=1) / max(med_d, 1e-12)
                cos = (x / np.maximum(xn, 1e-12)[:, None]) @ wnorm.T
                if grp == "word":
                    np.fill_diagonal(cos, -np.inf)     # not its own neighbour
                maha = np.sqrt((((x - mu) @ V) ** 2 / var).sum(1))
                rows.append({"layer": L, "group": grp, "n": int(len(x)),
                             "norm_z": round(float(np.median(z)), 3),
                             "centroid_ratio": round(float(np.median(ratio)), 3),
                             "max_cos": round(float(np.median(cos.max(1))), 4),
                             "maha_pca50": round(float(np.median(maha)), 2)})
        return rows

    summary = {}
    for fname in FRAMES:
        H = {g: np.stack(store[fname][g]) for g in store[fname]}
        summary[fname] = summarize(H)
        print(f"\n### frame {fname} ###")
        print(f"{'layer':>6}  " + "".join(f"{g:>26}" for g in
                                          ("inject", "trained", "randT")))
        print(f"{'':>6}  " + "".join(f"{'norm_z  ratio  maha':>26}" for _ in range(3)))
        by = {(r["layer"], r["group"]): r for r in summary[fname]}
        for L in range(n_layers):
            cells = ""
            for g in ("inject", "trained", "randT"):
                r = by[(L, g)]
                cells += f"{r['norm_z']:>9.2f}{r['centroid_ratio']:>8.2f}{r['maha_pca50']:>9.1f}"
            print(f"{L:>6}  {cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "model_key": args.model_key,
        "frames": FRAMES, "n_layers": n_layers, "hidden": hidden,
        "n_real_words": len(real), "n_trained": len(trained),
        "n_random": len(rand), "norm_trained_median": args.norm_trained,
        "measures": {
            "norm_z": "median ||h|| of the group, in SD of the word group's norms",
            "centroid_ratio": "median distance to the word centroid, over the "
                              "word group's own median distance to it",
            "max_cos": "median cosine to the nearest real word",
            "maha_pca50": "median Mahalanobis distance in the word cloud's top "
                          "50 principal components",
        },
        "summary": summary,
        "labels": labels,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
