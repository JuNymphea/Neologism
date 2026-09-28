#!/usr/bin/env python
"""Chinese Tasks 1-3 for a set of Chinese vectors run by submit_zh_spec_t123.slurm.

    python scripts/Tasks/controls/analyze_zh_spec.py --tag legacyzh --model gemma

Per template: calls per concept POS and accuracy for each task, the three-task
total (as in the template table), the unbiased noun/verb contrast, and the
template effect against unbiased (paired: P for Tasks 1 and 3, call counts for
Task 2). Task 3 tags Chinese synonyms with jieba and English ones (some vectors
answer in English) with WordNet (see analyze_zh_t13.py); a vector with no tagged
synonym is missing, not counted. Its adjective reading is unreliable.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, fisher_exact

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import analyze_zh_t13 as Z  # noqa: E402

T = HERE.parents[1]
CATS, POS = Z.CATS, Z.POS
TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gemma")
    a = ap.parse_args()
    Z.tag = Z.TaggerZH()
    m, tg = a.model, a.tag
    for way in (3, 2):
        cats = CATS if way == 3 else ("noun", "verb")
        cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11) if way == 3 or p != "a"]
        t2 = json.load(open(T / f"Task_2/out_zh/{tg}_vec{way}_calibration_{m}.json"))["neologisms"]
        D = {"T1": Z.t1(T / f"Task_1/out/{tg}_t1zh{way}_{m}.json", cats),
             "T2": {k: [float(v == c) for c in cats] for k, v in t2.items()},
             "T3": Z.t3(T / f"Task_3/out/{tg}_t3zh_synonyms_{m}.json", cats)}
        print(f"\n===== {m} zh ({tg}), {way}-way, {len(cons)} concepts: calls "
              + "/".join(cats) + " per concept POS (" + ", ".join(cats) + "), accuracy")
        for tp in TEMPLATES:
            line = f"{tp:9}"
            for task in ("T1", "T2", "T3"):
                h = {x: Z.hard(D[task][f"zh_{x}_{tp}"], cats) for x in cons}
                cell = " ".join("/".join(str(sum(h[x] == c for x in cons if POS[x[0]] == cp)) for c in cats) for cp in cats)
                ok = [x for x in cons if h[x] is not None]
                line += (f" | {task} {cell} acc {np.mean([h[x] == POS[x[0]] for x in ok]):.2f}"
                         + (f" [missing {len(cons) - len(ok)}]" if len(ok) < len(cons) else ""))
            print(line)
        if way == 3:
            print("three-task total (n=90): pred noun / verb / adj")
            for tp in ("noun", "verb", "adj", "unbiased", "mixed"):
                c = collections.Counter(Z.hard(D[t][f"zh_{x}_{tp}"], cats) for t in ("T1", "T2", "T3") for x in cons)
                print(f"  {tp:9} " + "  ".join(f"{c[k]} ({100 * c[k] / 90:.0f}%)" for k in cats))
        else:
            print("noun/verb contrast (noun concepts called noun - verb concepts called noun)")
            for tp in TEMPLATES:
                parts = []
                for task in ("T1", "T2", "T3"):
                    h = {x: Z.hard(D[task][f"zh_{x}_{tp}"], cats) for x in cons}
                    x1 = sum(h[x] == "noun" for x in cons if x[0] == "n"); x2 = sum(h[x] == "noun" for x in cons if x[0] == "v")
                    parts.append(f"{task} {10 * (x1 - x2):+d}pt (p={fisher_exact([[x1, 10 - x1], [x2, 10 - x2]])[1]:.2f})")
                print(f"  {tp:9} " + "  ".join(parts))
        print("template effect vs unbiased: P diff [Wilcoxon] for T1/T3, calls template vs unbiased for T2")
        for tp in ("noun", "verb", "adj"):
            if tp not in cats:
                continue
            k = cats.index(tp); parts = []
            for task in ("T1", "T3"):
                d = np.array([D[task][f"zh_{x}_{tp}"][k] - D[task][f"zh_{x}_unbiased"][k] for x in cons
                              if D[task][f"zh_{x}_{tp}"] is not None and D[task][f"zh_{x}_unbiased"] is not None])
                parts.append(f"{task} {d.mean():+.3f} (p={wilcoxon(d).pvalue if np.any(d) else 1:.3f})")
            n1 = sum(Z.hard(D["T2"][f"zh_{x}_{tp}"], cats) == tp for x in cons)
            n0 = sum(Z.hard(D["T2"][f"zh_{x}_unbiased"], cats) == tp for x in cons)
            print(f"  {tp:5}: " + "  ".join(parts) + f"  T2 {n1} vs {n0}")


if __name__ == "__main__":
    main()
