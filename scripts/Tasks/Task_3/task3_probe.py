#!/usr/bin/env python
"""Task 3: is lexical category linearly recoverable from the input embedding?

Task 1 reads what the model says a word is. Task 2 reads how it expects the
word to behave. Task 3 reads neither -- it asks whether the category is already
written into the lexical representation, before any Transformer layer, prompt,
continuation or generation step.

The claim being tested is *linear* recoverability, so the classifier is kept
deliberately weak: multinomial logistic regression on centred, L2-normalised
embeddings, one regularisation constant chosen on dev. A stronger classifier
would find structure that a linear read-out cannot use, which is not the claim.

Everything is fixed before any neologism is scored, in the same order Task 2
uses: fit on train, choose C on dev, report once on final-test, freeze. The
final-test words are `shared_finaltest`, the same hundred per category Task 2's
three-model run reported on, so the two probes compare word by word rather than
only in aggregate.

Three controls run alongside, because a linear probe reports a number whatever
you feed it:

  **Shuffled labels.** The same pipeline on permuted training labels, a hundred
  times. Anything above chance there is the pipeline, not the embeddings.

  **Morphology-reduced test.** The verb pool is 28.7% -ize/-ate/-ify against
  1.8% for nouns, so a probe could be reading derivational suffixes rather than
  category -- and a neologism is a single atomic token with no suffix at all.
  Accuracy is reported again on the test words that carry no overt cue.

  **Out-of-distribution check.** A learned neologism embedding can drift
  somewhere no pretrained word sits, where the probe extrapolates rather than
  classifies. Norm and distance to the lexical centroid are reported as
  z-scores against the known words.

    python task3_probe.py --model gemma
    python task3_probe.py --model gemma --neologisms run1=/path/embedding_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

POS_KEYS = ("noun", "verb", "adj")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

#: Suffixes that mark a category overtly. A probe leaning on these would not
#: transfer to a neologism, which is one atomic token.
OVERT = {
    "verb": ("ize", "ise", "ify", "ate"),
    "noun": ("ness", "ity", "tion", "sion", "ment"),
    "adj": ("ous", "ive", "al", "able", "ible", "ic"),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class Preprocess:
    """Centre on the training mean, then L2 normalise. Fitted on train only."""

    def __init__(self, X):
        self.mu = X.mean(0)

    def __call__(self, X):
        Z = X - self.mu
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(n, 1e-9)


def load(npz_path: Path, splits):
    d = np.load(npz_path, allow_pickle=True)
    vec = {w: v for w, v in zip(d["words"].tolist(), d["vectors"])}
    extra = {k[len("extra__"):]: d[k] for k in d.files if k.startswith("extra__")}
    out = {}
    for name, sp in splits.items():
        words = [w for p in POS_KEYS for w in sp[p]]
        y = np.array([POS_KEYS.index(p) for p in POS_KEYS for _ in sp[p]])
        out[name] = (words, np.stack([vec[w] for w in words]), y)
    return out, extra, int(d["dim"])


def fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000)
    clf.fit(X, y)
    return clf


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="模型简称，对应 out/embeddings/<name>.npz")
    ap.add_argument("--splits", type=Path, default=here / "out" / "task3_splits.json")
    ap.add_argument("--emb-dir", type=Path, default=here / "out" / "embeddings")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--neologisms", nargs="*", default=[], metavar="LABEL=PATH")
    ap.add_argument("--pos-keys", default="noun,verb,adj",
                    help="categories the probe is trained on; 'noun,verb' fits "
                         "a two-way probe and never sees adjectives")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global POS_KEYS
    POS_KEYS = tuple(k.strip() for k in args.pos_keys.split(",") if k.strip())
    out_path = args.out or (here / "out" / f"task3_{args.model}.json")

    splits = json.loads(args.splits.read_text())["splits"]
    data, extra, dim = load(args.emb_dir / f"{args.model}.npz", splits)
    (wtr, Xtr, ytr), (wdv, Xdv, ydv), (wte, Xte, yte) = (
        data["train"], data["dev"], data["finaltest"])
    print(f"{args.model}  {dim} 维   train {len(ytr)} / dev {len(ydv)} / test {len(yte)}")

    pre = Preprocess(Xtr)              # 只在 train 上拟合
    Ztr, Zdv, Zte = pre(Xtr), pre(Xdv), pre(Xte)

    # -- 1. C 只在 dev 上选 --------------------------------------------------
    print(f"\n正则化常数（只看 dev）")
    scores = []
    for C in C_GRID:
        acc = fit(Ztr, ytr, C).score(Zdv, ydv)
        scores.append((acc, C))
        print(f"  C={C:<7} dev {acc:.4f}")
    best_dev, C_star = max(scores)
    print(f"  选定 C = {C_star}（dev {best_dev:.4f}）")

    # -- 2. final-test 只报一次 ---------------------------------------------
    clf = fit(Ztr, ytr, C_star)
    pred = clf.predict(Zte)
    prob = clf.predict_proba(Zte)
    logit = clf.decision_function(Zte)
    # two classes give one signed score per word; mirror it so the
    # per-word records have one logit per class either way
    logit = np.asarray(logit)
    if logit.ndim == 1 and len(POS_KEYS) == 2:
        logit = np.stack([-logit, logit], axis=1)
    hit = int((pred == yte).sum())
    lo, hi = wilson(hit, len(yte))
    print(f"\n=== final-test（{len(yte)} 词，未参与任何选择）===")
    print(f"{'':8}{'n':>5}{'对':>5}{'准确率':>9}")
    per = {}
    for t, p in enumerate(POS_KEYS):
        m = yte == t
        per[p] = {"n": int(m.sum()), "correct": int((pred[m] == t).sum()),
                  "accuracy": float((pred[m] == t).mean())}
        print(f"  {p:<6}{per[p]['n']:>5}{per[p]['correct']:>5}{per[p]['accuracy']:>9.3f}")
    from sklearn.metrics import f1_score, confusion_matrix
    macro = f1_score(yte, pred, average="macro")
    print(f"  {'总体':<5}{len(yte):>5}{hit:>5}{hit/len(yte):>9.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  macro-F1 {macro:.3f}")
    cm = confusion_matrix(yte, pred)
    print(f"\n  混淆矩阵（行=真实，列=预测）\n{'':8}" + "".join(f"{p:>7}" for p in POS_KEYS))
    for t, p in enumerate(POS_KEYS):
        print(f"  {p:<6}" + "".join(f"{cm[t][j]:>7}" for j in range(len(POS_KEYS))))

    # -- 3. 随机标签对照 ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    sh = []
    for _ in range(args.shuffles):
        yp = rng.permutation(ytr)
        sh.append(fit(Ztr, yp, C_star).score(Zte, yte))
    sh = np.array(sh)
    print(f"\n=== 随机标签对照（{args.shuffles} 次）===")
    print(f"  准确率 均值 {sh.mean():.4f}  标准差 {sh.std():.4f}  最大 {sh.max():.4f}"
          f"   （理论基线 0.333）")
    print(f"  真实标签 {hit/len(yte):.4f} 超出随机对照最大值 "
          f"{'是' if hit/len(yte) > sh.max() else '否'}")

    # -- 4. 形态学对照 --------------------------------------------------------
    def overt(w, p):
        return any(w.endswith(s) for s in OVERT[p])
    keep = np.array([not overt(w, POS_KEYS[t]) for w, t in zip(wte, yte)])
    mr_hit = int((pred[keep] == yte[keep]).sum())
    mlo, mhi = wilson(mr_hit, int(keep.sum()))
    print(f"\n=== 形态学对照：去掉带显性派生后缀的测试词 ===")
    dropped = {p: sum(1 for w, t in zip(wte, yte) if POS_KEYS[t] == p and overt(w, p))
               for p in POS_KEYS}
    print(f"  剔除 {int((~keep).sum())}/{len(yte)} 个：{dropped}")
    print(f"  剩余 {int(keep.sum())} 个上的准确率 {mr_hit/max(keep.sum(),1):.3f}"
          f"   95% CI [{mlo:.3f}, {mhi:.3f}]")
    per_mr = {}
    for t, p in enumerate(POS_KEYS):
        m = (yte == t) & keep
        per_mr[p] = float((pred[m] == t).mean()) if m.any() else float("nan")
        print(f"    {p:<6}{int(m.sum()):>4} 个  {per_mr[p]:.3f}")

    # -- 5. 冻结，然后才看 neologism -----------------------------------------
    known_norm = np.linalg.norm(Xtr, axis=1)
    centroid = Xtr.mean(0)
    known_dist = np.linalg.norm(Xtr - centroid, axis=1)
    neo = {}
    for spec in args.neologisms:
        label, _, path = spec.partition("=")
        if path:
            import torch
            v = torch.load(path, map_location="cpu")
            # training saves the vector inside a payload (token, id, metadata),
            # so accept either that or a bare tensor
            if isinstance(v, dict):
                if "embedding" not in v:
                    raise SystemExit(f"{path} 里没有 embedding 字段：{sorted(v)[:6]}")
                v = v["embedding"]
            v = (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
        else:
            if label not in extra:
                raise SystemExit(f"npz 里没有 extra__{label}")
            v = extra[label]
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        z = pre(v[None, :])
        pr = fit_predict = clf.predict_proba(z)[0]
        lg = clf.decision_function(z)[0]
        # with two classes sklearn returns one signed score, not one per class
        lg = np.asarray(lg, dtype=float).reshape(-1)
        if lg.size == 1 and len(POS_KEYS) == 2:
            lg = np.array([-lg[0], lg[0]])
        neo[label] = {
            "probs": {p: float(pr[t]) for t, p in enumerate(POS_KEYS)},
            "logits": {p: float(lg[t]) for t, p in enumerate(POS_KEYS)},
            "predicted": POS_KEYS[int(pr.argmax())],
            "norm": float(np.linalg.norm(v)),
            "norm_z": float((np.linalg.norm(v) - known_norm.mean()) / known_norm.std()),
            "dist_to_centroid": float(np.linalg.norm(v - centroid)),
            "dist_z": float((np.linalg.norm(v - centroid) - known_dist.mean()) / known_dist.std()),
        }
    if neo:
        print(f"\n=== neologism（探针已冻结，不重新训练）===")
        print(f"{'':14}{'noun':>8}{'verb':>8}{'adj':>8}{'预测':>7}{'norm z':>9}{'dist z':>9}")
        for k, v in neo.items():
            print(f"  {k:<12}" + "".join(f"{v['probs'][p]:>8.3f}" for p in POS_KEYS)
                  + f"{v['predicted']:>7}{v['norm_z']:>9.2f}{v['dist_z']:>9.2f}")
        far = [k for k, v in neo.items() if abs(v["norm_z"]) > 5 or abs(v["dist_z"]) > 5]
        if far:
            print(f"  !! 偏离已知词分布 5 个标准差以上：{far} —— 这些预测是外推，要谨慎")

    out_path.write_text(json.dumps({
        "model": args.model, "dim": dim,
        "preprocessing": "centre on train mean, then L2 normalise",
        "classifier": "multinomial logistic regression, L2",
        "C_grid": list(C_GRID), "C_selected": C_star, "dev_accuracy": best_dev,
        "n": {"train": len(ytr), "dev": len(ydv), "finaltest": len(yte)},
        "finaltest": {"per_pos": per, "overall": hit / len(yte),
                      "ci95": [lo, hi], "macro_f1": float(macro),
                      "confusion": cm.tolist()},
        "shuffled_control": {"n": args.shuffles, "mean": float(sh.mean()),
                             "std": float(sh.std()), "max": float(sh.max())},
        "morphology_reduced": {"n": int(keep.sum()), "dropped": int((~keep).sum()),
                               "accuracy": mr_hit / max(int(keep.sum()), 1),
                               "ci95": [mlo, mhi], "per_pos": per_mr},
        "per_word": [{"word": w, "gold": POS_KEYS[t], "pred": POS_KEYS[int(q)],
                      "probs": {p: float(prob[i][j]) for j, p in enumerate(POS_KEYS)},
                      "logits": {p: float(logit[i][j]) for j, p in enumerate(POS_KEYS)}}
                     for i, (w, t, q) in enumerate(zip(wte, yte, pred))],
        "neologisms": neo,
        "probe": {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                  "mean": pre.mu.tolist()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
