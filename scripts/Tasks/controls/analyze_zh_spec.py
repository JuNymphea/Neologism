#!/usr/bin/env python
"""Chinese Tasks 1-3 for a set of Chinese vectors run by submit_zh_spec_t123.slurm.

    python scripts/Tasks/controls/analyze_zh_spec.py --tag legacyzh --model gemma

Per template: calls per concept POS and accuracy for each task, the three-task
total (as in the template table), the unbiased noun/verb contrast, and the
template effect against unbiased (paired: P for Tasks 1 and 3, call counts for
Task 2). Task 3 by default reads the LLM frame compatibility of the synonyms
(--t3 frames; ties split evenly), or jieba tags (--t3 jieba). A vector with no
readable synonym is missing, not counted.
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
_cache = None


def t3_frames(path, cats):
    """Frame compatibility restricted to cats, as a list (not normalised)."""
    global _cache
    import analyze_zh_frames as ZF
    if _cache is None:
        _cache = json.load(open(T / "Task_3/out/zh_frames_llm.json"))
    idx = [("noun", "verb", "adj").index(c) for c in cats]
    out = {k: (None if v is None else [float(v[i]) for i in idx]) for k, v in ZF.compat(path, _cache).items()}
    # ties left after the extra samples, settled by the synonyms' primary use
    # (Task_3/tiebreak_primary_zh.py): the decided category gets a hair more
    tb = T / "Task_3/out/zh_tiebreak_primary.json"
    dec = json.load(open(tb)).get(Path(path).stem, {}) if tb.exists() else {}
    way = "3" if len(cats) == 3 else "2"
    for k, v in out.items():
        c = (dec.get(k) or {}).get(way)
        if v is not None and c in cats:
            m = max(v)
            if sum(x >= m - 1e-9 for x in v) > 1:
                v[cats.index(c)] = m + 1e-6
    return out


def frac(v, cats):
    """Calls as weights: the top category gets 1, a tie is split evenly, missing gets nothing."""
    if v is None:
        return {}
    m = max(v)
    top = [c for c, x in zip(cats, v) if x >= m - 1e-9]
    return {c: 1 / len(top) for c in top}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--t3", default="frames", choices=("frames", "jieba"),
                    help="frames: the LLM frame compatibility (analyze_zh_frames.py), ties split "
                         "evenly; jieba: the dictionary tags of analyze_zh_t13.py")
    a = ap.parse_args()
    Z.tag = Z.TaggerZH()
    m, tg = a.model, a.tag
    for way in (3, 2):
        cats = CATS if way == 3 else ("noun", "verb")
        cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11) if way == 3 or p != "a"]
        t2 = json.load(open(T / f"Task_2/out_zh/{tg}_vec{way}_calibration_{m}.json"))["neologisms"]
        D = {"T1": Z.t1(T / f"Task_1/out/{tg}_t1zh{way}_{m}.json", cats),
             "T2": {k: [float(v == c) for c in cats] for k, v in t2.items()},
             "T3": (t3_frames(T / f"Task_3/out/{tg}_t3zh_synonyms_{m}.json", cats) if a.t3 == "frames"
                    else Z.t3(T / f"Task_3/out/{tg}_t3zh_synonyms_{m}.json", cats))}
        print(f"\n===== {m} zh ({tg}), {way}-way, {len(cons)} concepts: calls "
              + "/".join(cats) + " per concept POS (" + ", ".join(cats) + "), accuracy")
        for tp in TEMPLATES:
            line = f"{tp:9}"
            for task in ("T1", "T2", "T3"):
                h = {x: frac(D[task][f"zh_{x}_{tp}"], cats) for x in cons}
                cell = " ".join("/".join(f"{sum(h[x].get(c, 0) for x in cons if POS[x[0]] == cp):g}" for c in cats) for cp in cats)
                ok = [x for x in cons if h[x]]
                line += (f" | {task} {cell} acc {np.mean([h[x].get(POS[x[0]], 0) for x in ok]):.2f}"
                         + (f" [missing {len(cons) - len(ok)}]" if len(ok) < len(cons) else ""))
            print(line)
        if way == 3:
            print("three-task total (n=90): pred noun / verb / adj")
            for tp in ("noun", "verb", "adj", "unbiased", "mixed"):
                c = collections.Counter()
                for t in ("T1", "T2", "T3"):
                    for x in cons:
                        c.update(frac(D[t][f"zh_{x}_{tp}"], cats))
                print(f"  {tp:9} " + "  ".join(f"{c[k]:g} ({100 * c[k] / 90:.0f}%)" for k in cats))
        else:
            print("noun/verb contrast (noun concepts called noun - verb concepts called noun)")
            for tp in TEMPLATES:
                parts = []
                for task in ("T1", "T2", "T3"):
                    h = {x: frac(D[task][f"zh_{x}_{tp}"], cats) for x in cons}
                    x1 = sum(h[x].get("noun", 0) for x in cons if x[0] == "n"); x2 = sum(h[x].get("noun", 0) for x in cons if x[0] == "v")
                    r1, r2 = round(x1), round(x2)
                    parts.append(f"{task} {10 * (x1 - x2):+.0f}pt (p={fisher_exact([[r1, 10 - r1], [r2, 10 - r2]])[1]:.2f})")
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
            n1 = sum(frac(D["T2"][f"zh_{x}_{tp}"], cats).get(tp, 0) for x in cons)
            n0 = sum(frac(D["T2"][f"zh_{x}_unbiased"], cats).get(tp, 0) for x in cons)
            print(f"  {tp:5}: " + "  ".join(parts) + f"  T2 {n1:g} vs {n0:g}")


if __name__ == "__main__":
    main()
