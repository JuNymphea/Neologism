#!/usr/bin/env python
"""Task 3, read off the hidden states instead of the embedding row.

The embedding-row probe saturates, and the reason is measured rather than
guessed: at layer 0 the trained vectors sit +10.1 SD (gemma), +3.0 (qwen),
-1.8 (aya) from the real-word row norms, so a classifier fitted on those rows
extrapolates far outside its own data. L2-normalising repairs the norm and not
the direction.

Two findings from `controls/layer_drift.py` shape what this does instead.

**The norm drift dies at layer 1.** These are pre-norm models, the row meets an
RMSNorm before any weight, and gemma's trained group goes from z=+10.13 at layer
0 to z=-1.71 at layer 1. Layer 0 is the only place the wild extrapolation lives,
so the fix is to stop probing it.

**A real word's row injected into the token behaves exactly like the word.** The
`inject` group's distance to the word centroid is 1.00x the word group's own, at
every layer. So the probe is fitted on *injected* words -- the same token, the
same frame, the same position, with only the embedding row changing -- and train
and test conditions become identical by construction. That is what removes the
extrapolation, not the change of layer by itself.

What remains and is reported rather than hidden: the trained vectors still sit
about 2.0 to 2.6x further from the word centroid than a real word does in the
middle layers, and random vectors sit at the same distance. A probe could
therefore be reading "off-manifold" rather than "part of speech", so the nulls
are classified alongside and printed next to the trained vectors. If random
directions are labelled as confidently as trained vectors, the probe is reading
nothing, and this prints what that would look like.

The splits are Task 2's, used the same way: calibration words fit the
classifier, probe-dev picks the layer, final-test words report its accuracy, and
no trained vector touches any of the three.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: Category-neutral, and verified to address the right position. `the word "{W}"`
#: was dropped: after a quote the item tokenizes without its leading space.
FRAMES = {"list": "one , {W} , another", "bare": "{W}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--task2-out", type=Path, required=True,
                    help="where shared_calibration/probedev/finaltest.json live")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--norm-trained", type=float, required=True)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--pos-keys", default="noun,verb,adj")
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
    from train_neologism import _untie_output_embeddings  # noqa: E402

    pos_keys = [p.strip() for p in args.pos_keys.split(",") if p.strip()]
    short = {"noun": "n", "verb": "v", "adj": "a"}

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
    print(f"{args.new_token} -> id {new_id}; hidden {hidden}; {device}", flush=True)

    # ---- items ----------------------------------------------------------
    splits = {}
    for role in ("calibration", "probedev", "finaltest"):
        ws = json.loads((args.task2_out / f"shared_{role}.json").read_text())["words"]
        got = []
        for pos in pos_keys:
            for w in ws[pos]:
                ids = tok.encode(" " + w, add_special_tokens=False)
                if len(ids) == 1:
                    got.append((pos, w, ids[0]))
        splits[role] = got
        print(f"  {role}: {len(got)} single-token words", flush=True)

    trained = [(f"{r['lang']}_{r['concept']}_{r['template']}", r["best_file"])
               for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
               if r["model"] == args.model_key]
    print(f"  trained: {len(trained)}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    word_norm = float(np.median([float(emb[i].detach().float().norm())
                                 for _, _, i in splits["finaltest"]]))
    rand = []
    for tag, norm in (("randT", args.norm_trained), ("randW", word_norm)):
        for i in range(args.n_random):
            v = torch.empty(hidden).normal_(0.0, 1.0, generator=g)
            rand.append((f"{tag}_{i:03d}", v / v.norm() * norm))
    print(f"  random: {len(rand)} ({args.norm_trained:.3f} and {word_norm:.3f})",
          flush=True)

    # ---- collect hidden states, everything through the token slot --------
    n_layers = [None]
    original_row = emb[new_id].detach().clone()

    @torch.no_grad()
    def states(tmpl: str) -> np.ndarray:
        text = tmpl.replace("{W}", args.new_token)
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        head = tmpl.split("{W}")[0]
        n_pre = len(tok(head, add_special_tokens=True)["input_ids"])
        ids = enc["input_ids"][0]
        if not 0 <= n_pre < ids.shape[0]:
            raise SystemExit(f"frame prefix {head!r} -> position {n_pre} of "
                             f"{ids.shape[0]}: {text!r}")
        if int(ids[n_pre]) != new_id:
            raise SystemExit(f"position {n_pre} holds id {int(ids[n_pre])}, "
                             f"not {new_id}: {text!r}")
        out = model(**enc, output_hidden_states=True)
        n_layers[0] = len(out.hidden_states)
        return np.stack([h[0, n_pre].float().cpu().numpy()
                         for h in out.hidden_states])

    @torch.no_grad()
    def write_row(vec):
        emb[new_id] = vec.to(emb.dtype).to(emb.device)

    H: dict = {}
    for fname, tmpl in FRAMES.items():
        H[fname] = {}
        print(f"\n=== frame {fname}: {tmpl} ===", flush=True)
        for role in ("calibration", "probedev", "finaltest"):
            rows, ys = [], []
            for i, (pos, w, rid) in enumerate(splits[role], 1):
                write_row(emb[rid].detach().clone())
                rows.append(states(tmpl))
                ys.append(pos)
                if i % 200 == 0:
                    print(f"  {role} {i}/{len(splits[role])}", flush=True)
            H[fname][role] = (np.stack(rows), np.array(ys))
        rows, labs = [], []
        for i, (lab, path) in enumerate(trained, 1):
            write_row(torch.load(path, map_location="cpu")["embedding"])
            rows.append(states(tmpl))
            labs.append(lab)
            if i % 100 == 0:
                print(f"  trained {i}/{len(trained)}", flush=True)
        H[fname]["trained"] = (np.stack(rows), np.array(labs))
        rows, labs = [], []
        for lab, v in rand:
            write_row(v)
            rows.append(states(tmpl))
            labs.append(lab)
        H[fname]["random"] = (np.stack(rows), np.array(labs))
    write_row(original_row)
    nL = n_layers[0]
    print(f"\n{nL} layers per item", flush=True)

    # ---- fit per layer --------------------------------------------------
    def fit(Xtr, ytr, L):
        sc = StandardScaler().fit(Xtr[:, L, :])
        clf = LogisticRegression(max_iter=3000, C=args.C, multi_class="auto")
        clf.fit(sc.transform(Xtr[:, L, :]), ytr)
        return sc, clf

    result = {}
    for fname in FRAMES:
        Xc, yc = H[fname]["calibration"]
        Xd, yd = H[fname]["probedev"]
        Xf, yf = H[fname]["finaltest"]
        sweep = []
        # Layer 0 is included so the failure it causes is on the record, not
        # asserted: it is the layer the embedding-row probe was using.
        for L in range(nL):
            sc, clf = fit(Xc, yc, L)
            sweep.append({"layer": L,
                          "calibration": round(float(clf.score(sc.transform(Xc[:, L, :]), yc)), 4),
                          "probedev": round(float(clf.score(sc.transform(Xd[:, L, :]), yd)), 4)})
        best = max(sweep, key=lambda r: r["probedev"])
        L = best["layer"]
        sc, clf = fit(Xc, yc, L)

        def apply(X):
            P = clf.predict_proba(sc.transform(X[:, L, :]))
            return clf.predict(sc.transform(X[:, L, :])), P.max(1)

        pf, cf = apply(Xf)
        pt, ct = apply(H[fname]["trained"][0])
        pr, cr = apply(H[fname]["random"][0])
        tl, rl = H[fname]["trained"][1], H[fname]["random"][1]

        # distance to the fitting cloud, so the extrapolation is a number
        mu = Xc[:, L, :].mean(0)
        med = float(np.median(np.linalg.norm(Xc[:, L, :] - mu, axis=1)))
        dist = lambda X: round(float(np.median(
            np.linalg.norm(X[:, L, :] - mu, axis=1)) / max(med, 1e-9)), 3)

        import collections
        hit = float((pf == yf).mean())
        by_pos = {p: round(float((pf[yf == p] == p).mean()), 4) for p in pos_keys}
        match = float(np.mean([pt[i] == {v: k for k, v in short.items()}[tl[i].split("_")[1]]
                               for i in range(len(tl))]))
        result[fname] = {
            "layer_sweep": sweep, "chosen_layer": L,
            "finaltest_accuracy": round(hit, 4), "finaltest_per_pos": by_pos,
            "trained_predictions": {t: p for t, p in zip(tl.tolist(), pt.tolist())},
            "trained_distribution": dict(collections.Counter(pt.tolist())),
            "trained_concept_match": round(match, 4),
            "trained_mean_confidence": round(float(ct.mean()), 4),
            "random_distribution": {
                tag: dict(collections.Counter(
                    pr[[i for i, x in enumerate(rl) if x.startswith(tag)]].tolist()))
                for tag in ("randT", "randW")},
            "random_mean_confidence": round(float(cr.mean()), 4),
            "finaltest_mean_confidence": round(float(cf.mean()), 4),
            "centroid_ratio": {"finaltest": dist(Xf),
                               "trained": dist(H[fname]["trained"][0]),
                               "random": dist(H[fname]["random"][0])},
        }
        r = result[fname]
        print(f"\n### frame {fname} ###")
        print(f"  层扫描（在 calibration 上拟合，probe-dev 选层）：")
        for s in sweep:
            bar = " <- 选中" if s["layer"] == L else ""
            print(f"    L{s['layer']:>2}  calib {s['calibration']:.4f}  "
                  f"probe-dev {s['probedev']:.4f}{bar}")
        print(f"  第 {L} 层，final-test {len(yf)} 个词：{r['finaltest_accuracy']:.4f}  "
              + "  ".join(f"{p}:{v:.3f}" for p, v in by_pos.items()))
        print(f"  训练向量 {len(tl)} 个：分布 {r['trained_distribution']}  "
              f"概念 POS 匹配 {r['trained_concept_match']:.4f}")
        print(f"  零基线 randT {r['random_distribution']['randT']}  "
              f"randW {r['random_distribution']['randW']}")
        print(f"  平均置信度：真实词 {r['finaltest_mean_confidence']:.3f}  "
              f"训练向量 {r['trained_mean_confidence']:.3f}  "
              f"随机 {r['random_mean_confidence']:.3f}")
        print(f"  到拟合点云质心的相对距离：真实词 {r['centroid_ratio']['finaltest']}  "
              f"训练向量 {r['centroid_ratio']['trained']}  "
              f"随机 {r['centroid_ratio']['random']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "model_key": args.model_key,
        "pos_keys": pos_keys, "frames": FRAMES, "n_layers": nL, "hidden": hidden,
        "fitted_on": "real words' own embedding rows injected into the new token",
        "n_words": {r: len(v) for r, v in splits.items()},
        "n_trained": len(trained), "n_random_per_norm": args.n_random,
        "norm_trained_median": args.norm_trained,
        "norm_realword_median": word_norm,
        "C": args.C, "seed": args.seed,
        "results": result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
