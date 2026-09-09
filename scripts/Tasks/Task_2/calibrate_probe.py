#!/usr/bin/env python
"""Calibrate the Task 2 probe and decide which slots are usable.

Two different validities are at stake, and they are established by different
evidence:

    corpus purity        is this environment POS-selective in real language?
    control-word AUC     can the model actually feel that difference?

The first is settled by `run_induction.py`. This script settles the second.

Procedure, in order:

1. Fit, per slot, a matched and a non-matched surprisal distribution on the
   **calibration** words.
2. Score every slot's separation (AUC, Cohen's d) on the **calibration** words
   as well, and drop the slots that cannot tell matched from non-matched.
3. Only then, score each **validation** word by the Gaussian log-likelihood
   ratio
       r_i(w) = log p(s_i(w) | match) - log p(s_i(w) | non-match)
   and predict the category with the highest mean r over its slots.
4. Report accuracy on the validation words.

Two separations matter here, and mixing them up is easy.

Slots are dropped using **calibration** words, not validation words. AUC is
computed directly from the surprisal values and does not involve the fit, so it
is legitimate on the same words the distributions were fitted on. Selecting
slots by their validation AUC and then reporting validation accuracy would make
that accuracy optimistic -- the slot set would have been chosen on the very
data it is scored on.

And the whole thing happens before any neologism is scored: choosing slots by
their effect on the main result would be probe tuning.

    python Task_2/calibrate_probe.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

POS_KEYS = ("noun", "verb", "adj")
UPPER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return -0.5 * math.log(2 * math.pi * sigma * sigma) - (x - mu) ** 2 / (2 * sigma * sigma)


def auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Probability a matched word scores lower (less surprising) than a non-matched one.

    Computed by rank sum rather than by thresholding, so it needs no
    distributional assumption and is comparable across slots.
    """
    if not pos_scores or not neg_scores:
        return float("nan")
    merged = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores])
    ranks, i = {}, 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks.setdefault(k, avg)
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n1, n0 = len(pos_scores), len(neg_scores)
    u = r_pos - n1 * (n1 + 1) / 2
    # Lower surprisal should mean "matched", so invert.
    return 1.0 - u / (n1 * n0)


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = st.stdev(a), st.stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2)
                       / (len(a) + len(b) - 2))
    return (st.mean(b) - st.mean(a)) / max(pooled, 1e-9)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surprisal", type=Path, default=here / "out" / "surprisal.json")
    ap.add_argument("--out", type=Path, default=here / "out" / "calibration.json")
    ap.add_argument("--min-auc", type=float, default=0.60,
                    help="slots below this on validation words are dropped")
    args = ap.parse_args()

    data = json.loads(args.surprisal.read_text())
    S = data["surprisal"]
    words = data["words"]
    slot_meta = {f"{s['pos']}::{s['signature']}": s for s in data["slots"]}

    calib = {p: words["calibration"][p] for p in POS_KEYS}
    valid = {p: words["validation"][p] for p in POS_KEYS}

    # -- 1. fit reference distributions on calibration words only -----------
    fits: Dict[str, dict] = {}
    for key, scores in S.items():
        pos = key.split("::")[0].lower()
        matched = [scores[w] for w in calib[pos] if w in scores
                   and not math.isnan(scores[w])]
        nonmatched = [scores[w] for p in POS_KEYS if p != pos
                      for w in calib[p] if w in scores and not math.isnan(scores[w])]
        if len(matched) < 2 or len(nonmatched) < 2:
            continue
        fits[key] = {
            "pos": pos,
            "mu_match": st.mean(matched), "sigma_match": st.stdev(matched),
            "mu_nonmatch": st.mean(nonmatched), "sigma_nonmatch": st.stdev(nonmatched),
            "n_match": len(matched), "n_nonmatch": len(nonmatched),
        }

    # -- 2. per-slot diagnostics -------------------------------------------
    # On the calibration words: selecting slots by their validation AUC would
    # make the validation accuracy that follows optimistic. The validation AUC
    # is computed too, but only reported -- never used to drop anything.
    diagnostics: Dict[str, dict] = {}
    for key, f in fits.items():
        pos = f["pos"]

        def split(pool):
            m = [S[key][w] for w in pool[pos]
                 if w in S[key] and not math.isnan(S[key][w])]
            n = [S[key][w] for p in POS_KEYS if p != pos for w in pool[p]
                 if w in S[key] and not math.isnan(S[key][w])]
            return m, n

        cm, cn = split(calib)
        vm, vn = split(valid)
        diagnostics[key] = {
            "auc": auc(cm, cn),                       # decides retention
            "cohens_d": cohens_d(cm, cn),
            "auc_validation": auc(vm, vn),            # reported only
            "mean_match": st.mean(cm) if cm else float("nan"),
            "mean_nonmatch": st.mean(cn) if cn else float("nan"),
            "example": slot_meta[key]["example"],
            "diagnostic": slot_meta[key]["diagnostic"],
        }

    kept = {k for k, d in diagnostics.items()
            if not math.isnan(d["auc"]) and d["auc"] >= args.min_auc}
    dropped = sorted(set(diagnostics) - kept)

    print(f"{'slot':<46}{'AUC':>7}{'d':>7}{'AUC_val':>9}  例句")
    print("=" * 110)
    for pos in POS_KEYS:
        for key in sorted((k for k in diagnostics if fits[k]["pos"] == pos),
                          key=lambda k: -diagnostics[k]["auc"]):
            d = diagnostics[key]
            flag = " " if key in kept else "✗"
            print(f"{flag}{key.split('::')[1]:<45}{d['auc']:>7.3f}{d['cohens_d']:>7.2f}"
                  f"{d['auc_validation']:>9.3f}  {d['example'][:34]}")
    print("=" * 110)
    print(f"保留 {len(kept)}/{len(diagnostics)} 个 slot"
          + (f"，淘汰 {len(dropped)} 个 (校准集 AUC < {args.min_auc})" if dropped else ""))
    print("AUC 由校准词决定去留；AUC_val 仅供参考，不参与任何决策。")

    # -- 3. three-way accuracy on validation words --------------------------
    def classify(word: str, slot_keys) -> str | None:
        means = {}
        for p in POS_KEYS:
            rs = []
            for key in slot_keys:
                if fits[key]["pos"] != p or word not in S[key]:
                    continue
                s = S[key][word]
                if math.isnan(s):
                    continue
                f = fits[key]
                rs.append(gaussian_logpdf(s, f["mu_match"], f["sigma_match"])
                          - gaussian_logpdf(s, f["mu_nonmatch"], f["sigma_nonmatch"]))
            if rs:
                means[p] = sum(rs) / len(rs)
        return max(means, key=means.get) if means else None

    results = {}
    for label, keys in (("all_slots", set(diagnostics)), ("kept_slots", kept)):
        per_pos, correct, total = {}, 0, 0
        for p in POS_KEYS:
            hits = sum(1 for w in valid[p] if classify(w, keys) == p)
            per_pos[p] = {"n": len(valid[p]), "correct": hits,
                          "accuracy": round(hits / len(valid[p]), 4)}
            correct += hits
            total += len(valid[p])
        results[label] = {"per_pos": per_pos,
                          "overall": round(correct / total, 4),
                          "n_slots": len(keys)}

    print(f"\n验证集三分类准确率（{sum(len(valid[p]) for p in POS_KEYS)} 个词，"
          f"既未参与拟合，也未参与 slot 淘汰）")
    print(f"{'':14}{'noun':>8}{'verb':>8}{'adj':>8}{'overall':>10}{'slots':>7}")
    for label, r in results.items():
        print(f"{label:<14}" + "".join(f"{r['per_pos'][p]['accuracy']:>8.3f}"
                                       for p in POS_KEYS)
              + f"{r['overall']:>10.3f}{r['n_slots']:>7}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": data.get("model"),
        "min_auc": args.min_auc,
        "auc_computed_on": "calibration words (validation kept clean for accuracy)",
        "fits": fits,
        "diagnostics": diagnostics,
        "kept_slots": sorted(kept),
        "dropped_slots": dropped,
        "accuracy": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
