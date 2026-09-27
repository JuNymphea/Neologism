#!/usr/bin/env python
"""gemma en on the fn30 data: Tasks 1-3 for every template.

fn30 is the 30-concept set taken from the earlier FrameNet data (short concept
answers, scripts/train/data_fn30), renumbered n/v/a_1..10 -- its ids do not
match the current set's. Vectors: seed 42, hinge 0.1, 1 epoch, raw prompt.

    python scripts/Tasks/controls/analyze_fn30.py [--tag fn30] [--model gemma]

Reported per template and task, three-way and two-way (noun/verb concepts,
Task 3 synonym tags renormalized over noun and verb):
  * what each concept POS is called, and accuracy;
  * two-way contrast: noun concepts called noun minus verb concepts called noun;
  * the template effect: P(template's POS) under the template minus under
    unbiased, same concept, paired Wilcoxon (Task 1 and 3 probabilities), and
    the count of calls for the template's POS (Task 2 is hard calls only).
Writes scripts/Tasks/task123_tables/<tag>_<model>_en_stats.csv.
"""
import argparse, collections, csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyze_seeds import t1, t2, t3, hard, CATS  # noqa: E402

T = HERE.parents[1]
TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")
POS = {"n": "noun", "v": "verb", "a": "adj"}


def load(tag, model, way):
    cats = CATS if way == 3 else ("noun", "verb")
    out = {}
    for tp in TEMPLATES:
        run = f"{tag}_{model}_{tp}"
        out[tp] = {"T1": t1(T / f"Task_1/out/{run}_{way}.json", cats),
                   "T2": t2(T / f"Task_2/out/{run}_{way}_calibration.json"),
                   "T3": t3(T / f"Task_3/out/{run}_synonyms.json", cats)}
    return cats, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="fn30")
    ap.add_argument("--model", default="gemma")
    args = ap.parse_args()
    rows = []
    for way in (3, 2):
        cats, D = load(args.tag, args.model, way)
        cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11) if way == 3 or p != "a"]
        print(f"\n===== {args.model} en, {args.tag}, {'three' if way == 3 else 'two'}-way, {len(cons)} concepts =====")
        print("calls " + "/".join(cats) + " per concept POS; accuracy")
        for tp in TEMPLATES:
            line = f"{tp:9}"
            for task in ("T1", "T2", "T3", "sum"):
                calls = {}
                for x in cons:
                    if task == "sum":
                        continue
                    calls[x] = hard(D[tp][task][f"en_{x}_{tp}"], cats)
                if task == "sum":
                    by = collections.Counter()
                    acc_n = acc_d = 0
                    for tk in ("T1", "T2", "T3"):
                        for x in cons:
                            c = hard(D[tp][tk][f"en_{x}_{tp}"], cats)
                            by[(POS[x[0]], c)] += 1
                            acc_n += c == POS[x[0]]; acc_d += 1
                else:
                    by = collections.Counter((POS[x[0]], calls[x]) for x in cons)
                    acc_n = sum(calls[x] == POS[x[0]] for x in cons); acc_d = len(cons)
                cell = " ".join("/".join(str(by[(cp, c)]) for c in cats) for cp in cats)
                line += f" | {task} {cell} acc {acc_n / acc_d:.2f}"
                for cp in cats:
                    rows.append(dict(model=args.model, lang="en", data=args.tag, template=tp, task=task,
                                     classes=f"{way}way", concept_pos=cp,
                                     n=sum(by[(cp, c)] for c in cats),
                                     **{f"pred_{c}": by[(cp, c)] for c in CATS},
                                     accuracy=round(acc_n / acc_d, 3)))
            print(line)
        if way == 2:
            print("two-way contrast (noun concepts called noun - verb concepts called noun), pt")
            for tp in TEMPLATES:
                vals = []
                for task in ("T1", "T2", "T3"):
                    h = {x: hard(D[tp][task][f"en_{x}_{tp}"], cats) for x in cons}
                    n = np.mean([h[x] == "noun" for x in cons if x[0] == "n"])
                    v = np.mean([h[x] == "noun" for x in cons if x[0] == "v"])
                    vals.append(f"{task} {100 * (n - v):+5.0f}")
                print(f"  {tp:9} " + "  ".join(vals))
        print("template effect vs unbiased: P(template POS) difference [Wilcoxon p] (T1, T3); "
              "calls for template POS, template vs unbiased (T2)")
        for tp in ("noun", "verb", "adj", "mixed"):
            for target in ([tp] if tp != "mixed" else list(cats)):
                if target not in cats:
                    continue
                k = cats.index(target)
                parts = []
                for task in ("T1", "T3"):
                    d = np.array([D[tp][task][f"en_{x}_{tp}"][k] - D["unbiased"][task][f"en_{x}_unbiased"][k]
                                  for x in cons])
                    p = wilcoxon(d).pvalue if np.any(d) else 1.0
                    parts.append(f"{task} {d.mean():+.3f} [p={p:.3f}]")
                a = sum(hard(D[tp]["T2"][f"en_{x}_{tp}"], cats) == target for x in cons)
                b = sum(hard(D["unbiased"]["T2"][f"en_{x}_unbiased"], cats) == target for x in cons)
                parts.append(f"T2 {a} vs {b}")
                print(f"  {tp:6} -> {target:5}: " + "  ".join(parts))
    out = T / "task123_tables" / f"{args.tag}_{args.model}_en_stats.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
