#!/usr/bin/env python
"""Verb-template vectors from the original training code vs the current code.

Same fn30 data, same 30 concepts, same seed 42; "legacy" is
scripts/train/legacy/train_neologism_legacy.py, "current" the fn30 sweep.
The current code's unbiased vectors are the reference the template moves from.

Per task: calls noun/verb/adj over the 30 concepts, accuracy, mean P(verb),
paired differences, plus Task 1 as the original code scored it (T1old) and the
three-task total in the format of the earlier template table.
"""
import collections, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyze_seeds import t1, t2, t3, hard, CATS  # noqa: E402

T = HERE.parents[1]
POS = {"n": "noun", "v": "verb", "a": "adj"}


def t1old(path, label_suffix):
    neo = json.load(open(path))["neologism"]
    return {k: [v["probabilities"][c] for c in CATS] for k, v in neo.items() if k.endswith(label_suffix)}


def main():
    cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11)]
    o1, o2, o3 = T / "Task_1/out", T / "Task_2/out", T / "Task_3/out"
    sets = {
        "legacy verb": ({"T1": t1(o1 / "legacy_gemma_verb_3.json", CATS),
                         "T1old": t1old(o1 / "legacy_legacytrain_gemma_verb.json", "_verb"),
                         "T2": t2(o2 / "legacy_gemma_verb_3_calibration.json"),
                         "T3": t3(o3 / "legacy_gemma_verb_synonyms.json", CATS)}, "verb"),
    }
    old_fn30 = o1 / "legacy_fn30_gemma.json"
    for tp in ("verb", "unbiased"):
        sets[f"current {tp}"] = ({"T1": t1(o1 / f"fn30_gemma_{tp}_3.json", CATS),
                                  "T1old": t1old(old_fn30, f"_{tp}"),
                                  "T2": t2(o2 / f"fn30_gemma_{tp}_3_calibration.json"),
                                  "T3": t3(o3 / f"fn30_gemma_{tp}_synonyms.json", CATS)}, tp)
    tasks = ("T1", "T1old", "T2", "T3")
    print("calls noun/verb/adj (30 concepts), accuracy, mean P(verb)")
    for name, (src, tp) in sets.items():
        line = f"{name:15}"
        for task in tasks:
            h = [hard(src[task][f"en_{x}_{tp}"], CATS) for x in cons]
            c = collections.Counter(h)
            acc = np.mean([a == POS[x[0]] for a, x in zip(h, cons)])
            pv = "" if task == "T2" else f" P(v) {np.mean([src[task][f'en_{x}_{tp}'][1] for x in cons]):.2f}"
            line += f" | {task} {c['noun']}/{c['verb']}/{c['adj']} acc {acc:.2f}{pv}"
        print(line)
    print("\nthree-task total (T1 few-shot, T2, T3), as in the earlier template table; n=90")
    for name, (src, tp) in sets.items():
        c = collections.Counter(hard(src[task][f"en_{x}_{tp}"], CATS) for task in ("T1", "T2", "T3") for x in cons)
        print(f"  {name:15} " + "  ".join(f"pred={k} {c[k]} ({100 * c[k] / 90:.0f}%)" for k in CATS))
    print("\npaired P(verb) differences")
    for task in ("T1", "T1old", "T3"):
        for a, b in (("legacy verb", "current verb"), ("legacy verb", "current unbiased"),
                     ("current verb", "current unbiased")):
            A, ta = sets[a]; B, tb = sets[b]
            d = np.array([A[task][f"en_{x}_{ta}"][1] - B[task][f"en_{x}_{tb}"][1] for x in cons])
            print(f"  {task:5} {a} - {b}: {d.mean():+.3f} (Wilcoxon p={wilcoxon(d).pvalue:.3f})")
    print("\nagreement of hard calls, legacy verb vs current verb, same concept")
    A, _ = sets["legacy verb"]; B, _ = sets["current verb"]
    print("  " + "  ".join(f"{task} {np.mean([hard(A[task][f'en_{x}_verb'], CATS) == hard(B[task][f'en_{x}_verb'], CATS) for x in cons]):.2f}"
                           for task in tasks))


if __name__ == "__main__":
    main()
