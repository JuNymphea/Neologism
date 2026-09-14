#!/usr/bin/env python
"""Test Task 1 on the Task 2 control words, and compare the two probes.

Task 1 asks the model outright; Task 2 reads syntactic behaviour off
continuation surprisal. Run on the same words, they answer different
questions about the same model, and the gap between them is the interesting
quantity: a word the model can *use* as a verb but cannot *name* as one is
exactly the dissociation the paper is about.

Two things this script does that `run_task1.py` does not:

**It uses all three Task 2 splits, not just final-test.** Task 1 fits nothing
and selects nothing, so calibration and probe-dev were never spent on it --
those 222 extra words per category are as clean for Task 1 as final-test is.
That triples the sample and roughly halves the confidence interval. The splits
are still reported separately, because only the final-test column is
comparable to Task 2's headline number; the combined column is Task 1's own
best estimate.

**It compares the two probes item by item.** Overall accuracies can match while
the probes agree on nothing, so agreement rate and McNemar's test are reported
alongside. McNemar is the right test here because the two probes are measured
on the *same* words: what matters is the words where they disagree, not the
marginal totals.

    python Task_1/eval_on_task2_words.py --model google/gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

from run_task1 import (CASINGS, POS_KEYS, accuracy_report, aggregate,
                       build_variants, load_model, score_log_probs)

SPLITS = ("calibration", "probedev", "finaltest")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval -- behaves at the extremes, where normal fails."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return (-0.5 * math.log(2 * math.pi * sigma * sigma)
            - (x - mu) ** 2 / (2 * sigma * sigma))


def task2_predictions(surprisal: Path, calibration: Path) -> Dict[str, str]:
    """Task 2's per-word prediction, rebuilt from its frozen fits.

    `calibration.json` reports accuracy but not which word went where, so the
    log-likelihood ratio is recomputed here -- same formula as
    `calibrate_probe.py`, on the same frozen slot set.
    """
    S = json.loads(surprisal.read_text())["surprisal"]
    cal = json.loads(calibration.read_text())
    fits = cal["fits"]
    keys = [k for k in cal["kept_slots"] if k in fits and k in S]

    preds: Dict[str, str] = {}
    for word in {w for k in keys for w in S[k]}:
        means = {}
        for p in POS_KEYS:
            rs = []
            for k in keys:
                if fits[k]["pos"] != p:
                    continue
                s = S[k].get(word)
                if s is None or math.isnan(s):
                    continue
                f = fits[k]
                rs.append(gaussian_logpdf(s, f["mu_match"], f["sigma_match"])
                          - gaussian_logpdf(s, f["mu_nonmatch"], f["sigma_nonmatch"]))
            if rs:
                means[p] = sum(rs) / len(rs)
        if means:
            preds[word] = max(means, key=means.get)
    return preds


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p for the discordant pairs.

    Exact rather than chi-square: the discordant counts here are small enough
    that the continuity-corrected approximation is unreliable.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    here = Path(__file__).resolve().parent
    task2 = here.parent / "Task_2" / "out"
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--task2-out", type=Path, default=task2,
                    help="directory holding the Task 2 control-word files")
    ap.add_argument("--out", type=Path, default=here / "out" / "task1_on_task2_words.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-articles", action="store_true")
    ap.add_argument("--casings", default="lower")
    ap.add_argument("--no-abbrev", action="store_true")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the Task 2 comparison even if its outputs exist")
    args = ap.parse_args()

    cases = tuple(c.strip() for c in args.casings.split(",") if c.strip())
    if [c for c in cases if c not in CASINGS]:
        ap.error(f"--casings 只接受 {', '.join(CASINGS)}")
    variants = build_variants(not args.no_articles, cases, not args.no_abbrev)
    forms = sorted({v for c in POS_KEYS for v in variants[c]})

    # -- the words -----------------------------------------------------------
    splits: Dict[str, Dict[str, List[str]]] = {}
    for name in SPLITS:
        path = args.task2_out / f"control_{name}.json"
        if not path.exists():
            ap.error(f"缺少 {path}；先跑 Task_2/build_control_words.py")
        splits[name] = json.loads(path.read_text())["words"]

    # A word could in principle appear in two splits; that would double-count
    # it in the combined column, so check rather than assume.
    seen: Counter = Counter(w for s in splits.values() for p in POS_KEYS for w in s[p])
    dupes = [w for w, n in seen.items() if n > 1]
    if dupes:
        print(f"警告：{len(dupes)} 个词跨 split 重复，合并列会重复计数：{dupes[:5]}")

    print(f"变体集：" + "  ".join(f"{c}={len(variants[c])}" for c in POS_KEYS)
          + f"，共 {len(forms)} 个表层形式")
    for name in SPLITS:
        print(f"  {name:<12} " + ", ".join(f"{p}={len(splits[name][p])}"
                                           for p in POS_KEYS))
    total_words = sum(len(splits[s][p]) for s in SPLITS for p in POS_KEYS)
    print(f"合计 {total_words} 个词 x {len(forms)} 个形式 = "
          f"{total_words * len(forms):,} 次 teacher forcing")

    tok, model, device = load_model(args.model, args.device, args.dtype)

    # -- score ---------------------------------------------------------------
    logps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in SPLITS:
        for pos in POS_KEYS:
            lp = score_log_probs(tok, model, device, splits[name][pos], forms,
                                 args.batch_size, label=f"{name}/{pos}")
            logps.setdefault(name, {})[pos] = lp

    probs = {name: {pos: {w: aggregate(lp, variants) for w, lp in d.items()}
                    for pos, d in by_pos.items()}
             for name, by_pos in logps.items()}

    # -- accuracy per split, then combined -----------------------------------
    reports = {name: accuracy_report(probs[name], f"Task 1 · {name}")
               for name in SPLITS}
    combined_probs = {pos: {w: pr for name in SPLITS
                            for w, pr in probs[name][pos].items()}
                      for pos in POS_KEYS}
    reports["combined"] = accuracy_report(
        combined_probs, "Task 1 · 三个 split 合并（Task 1 不拟合，全部都是干净的）")

    print(f"\n各 split 对比（Wilson 95% CI）\n{'':14}{'n':>6}{'准确率':>10}"
          f"{'95% CI':>20}")
    for name in [*SPLITS, "combined"]:
        r = reports[name]
        n = sum(r[p]["n"] for p in POS_KEYS)
        k = sum(r[p]["correct"] for p in POS_KEYS)
        lo, hi = wilson(k, n)
        print(f"{name:<14}{n:>6}{k / n:>10.3f}      [{lo:.3f}, {hi:.3f}]")
    print("只有 finaltest 一行能和 Task 2 的数字直接比；combined 是 Task 1 自己"
          "最准的估计。")

    # -- item-level comparison with Task 2 -----------------------------------
    comparison: dict = {}
    surp, calib = args.task2_out / "surprisal.json", args.task2_out / "calibration.json"
    if args.no_compare:
        pass
    elif not (surp.exists() and calib.exists()):
        print(f"\n没找到 {surp.name} / {calib.name}，跳过与 Task 2 的对比。")
    else:
        t2 = task2_predictions(surp, calib)
        rows = [(w, pos, max(pr, key=pr.get), t2[w])
                for pos in POS_KEYS
                for w, pr in probs["finaltest"][pos].items() if w in t2]
        if not rows:
            print("\nTask 2 的预测里没有 final-test 词，跳过对比。")
        else:
            both = sum(a == t and b == t for _, t, a, b in rows)
            only1 = sum(a == t and b != t for _, t, a, b in rows)
            only2 = sum(a != t and b == t for _, t, a, b in rows)
            neither = sum(a != t and b != t for _, t, a, b in rows)
            agree = sum(a == b for _, _, a, b in rows)
            p = mcnemar(only1, only2)
            n = len(rows)
            print(f"\n与 Task 2 逐词对比（final-test，{n} 个词）")
            print(f"{'':22}{'Task2 对':>10}{'Task2 错':>10}")
            print(f"{'Task1 对':<20}{both:>10}{only1:>10}")
            print(f"{'Task1 错':<20}{only2:>10}{neither:>10}")
            print(f"\n两者预测一致率 {agree / n:.3f}"
                  f"（{agree}/{n}，含一致地答错的 {neither} 个）")
            print(f"McNemar 精确检验 p = {p:.4f}"
                  + ("  —— 两个 probe 的准确率差异不显著"
                     if p >= 0.05 else "  —— 差异显著"))
            print(f"只有 Task 1 答对 {only1} 个，只有 Task 2 答对 {only2} 个，"
                  f"两个都错 {neither} 个")
            if neither:
                hard = [w for w, t, a, b in rows if a != t and b != t]
                print(f"两个 probe 都答错的词（{len(hard)} 个）：{hard[:20]}")
            comparison = {
                "n": n, "both": both, "only_task1": only1,
                "only_task2": only2, "neither": neither,
                "agreement": round(agree / n, 4),
                "mcnemar_p": round(p, 6),
                "per_word": [{"word": w, "gold": t, "task1": a, "task2": b}
                             for w, t, a, b in rows],
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "variants": variants,
        "splits": {n: {p: len(splits[n][p]) for p in POS_KEYS} for n in SPLITS},
        "accuracy": reports,
        "comparison_with_task2": comparison,
        "probabilities": probs,
        "log_probabilities": logps,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
