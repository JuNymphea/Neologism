#!/usr/bin/env python
"""Verb-template vectors on the current data: original training code vs current.

Same data (scripts/train/data/train/en), same 30 concepts; "legacy" is
train_neologism_legacy.py at seed 42, "current" the vectors every earlier table
used (vectors_best: 26 seed-42 1-epoch runs and 4 others) and the current
code's unbiased vectors as the reference. Holding the data fixed isolates the
training code; the fn30 comparison (analyze_legacy.py) changed data and code at
once.
"""
import collections, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyze_seeds import t1, t2, t3, hard, CATS  # noqa: E402

T = HERE.parents[1]
POS = {"n": "noun", "v": "verb", "a": "adj"}


def main():
    cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11)]
    o1, o2, o3 = T / "Task_1/out", T / "Task_2/out", T / "Task_3/out"
    cur = {"T1": t1(o1 / "fsvec3_gemma.json", CATS), "T2": t2(o2 / "neo_calibration_gemma.json"),
           "T3": t3(o3 / "synonyms_gemma.json", CATS)}
    sets = {
        "legacy verb": ({"T1": t1(o1 / "legacynew_gemma_verb_3.json", CATS),
                         "T2": t2(o2 / "legacynew_gemma_verb_3_calibration.json"),
                         "T3": t3(o3 / "legacynew_gemma_verb_synonyms.json", CATS)}, "verb"),
        "current verb": (cur, "verb"),
        "current unbiased": (cur, "unbiased"),
    }
    print("calls noun/verb/adj over 30 concepts, accuracy, mean P(verb)")
    for name, (src, tp) in sets.items():
        line = f"{name:17}"
        for task in ("T1", "T2", "T3"):
            h = [hard(src[task][f"en_{x}_{tp}"], CATS) for x in cons]
            c = collections.Counter(h)
            acc = np.mean([a == POS[x[0]] for a, x in zip(h, cons)])
            pv = "" if task == "T2" else f" P(v) {np.mean([src[task][f'en_{x}_{tp}'][1] for x in cons]):.2f}"
            line += f" | {task} {c['noun']}/{c['verb']}/{c['adj']} acc {acc:.2f}{pv}"
        print(line)
    print("\nthree-task total, n=90")
    for name, (src, tp) in sets.items():
        c = collections.Counter(hard(src[t][f"en_{x}_{tp}"], CATS) for t in ("T1", "T2", "T3") for x in cons)
        print(f"  {name:17} " + "  ".join(f"pred={k} {c[k]} ({100 * c[k] / 90:.0f}%)" for k in CATS))
    print("\npaired P(verb) differences")
    for task in ("T1", "T3"):
        for a, b in (("legacy verb", "current verb"), ("legacy verb", "current unbiased"),
                     ("current verb", "current unbiased")):
            A, ta = sets[a]; B, tb = sets[b]
            d = np.array([A[task][f"en_{x}_{ta}"][1] - B[task][f"en_{x}_{tb}"][1] for x in cons])
            print(f"  {task} {a} - {b}: {d.mean():+.3f} (Wilcoxon p={wilcoxon(d).pvalue:.3f})")


if __name__ == "__main__":
    main()
