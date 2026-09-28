#!/usr/bin/env python
"""English, verb template: fresh seeds against the vectors already on file.

    python scripts/Tasks/controls/analyze_seeds.py --model gemma --seeds 48 49
    python scripts/Tasks/controls/analyze_seeds.py --model qwen --seeds 48

Each new seed changes the N(0, 0.02) init and the data order; everything else (hinge
0.1, 1 epoch, raw prompt) matches the sweep. The question is whether gemma's
verb template reading as noun is a property of the seed-42 run or of the setup.

Compared, per concept (30, or the 20 noun/verb concepts for two-way):
  seedN verb        the new vectors, one set per seed
  best verb         vectors_best, what every table so far used: mostly seed-42
                    1-epoch runs (for gemma all but a_1 n_6 n_9 v_1; the manifest
                    says which), compared on those alone as well
  no-hinge verb     seed 42, lambda_h 0 (gemma only)
  best unbiased     seed-42 1-epoch unbiased, the reference the template moves from
Two-way Task 3 renormalizes the synonym tags over noun and verb.
"""
import argparse, itertools, json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "Task_3"))
from analyze_synonyms import parse, Tagger, CATS  # noqa: E402
tag = Tagger()
POS = {"n": "noun", "v": "verb", "a": "adj"}
MODEL = "gemma"


def not_seed42(model):
    """Concepts whose vectors_best verb vector is not a seed-42 1-epoch run."""
    import csv, torch
    root = T.parents[1]
    out = set()
    for r in csv.DictReader(open(root / "scripts/train/vectors_best/manifest.csv")):
        if r["model"] == model and r["lang"] == "en" and r["template"] == "verb":
            v = torch.load(root / r["best_file"], map_location="cpu")
            if v.get("seed") != 42 or str(r["epochs"]) != "1":
                out.add(r["concept"])
    return out


def t1(path, cats):
    return {k: [v[c] for c in cats] for k, v in json.load(open(path))["probabilities"]["neologism"].items()}


def t2(path):
    return json.load(open(path))["neologisms"]


def t3(path, cats):
    out = {}
    for k, samples in json.load(open(path))["results"].items():
        acc, w = np.zeros(3), 0
        for s in samples:
            for x in parse(s["raw"]):
                d = tag(x)
                if d:
                    acc += [d[c] for c in CATS]; w += 1
        p = acc / w if w else np.ones(3) / 3
        p = np.array([p[CATS.index(c)] for c in cats])
        out[k] = list(p / p.sum()) if p.sum() else [1 / len(cats)] * len(cats)
    return out


def load(way, seeds):
    cats = CATS if way == 3 else ("noun", "verb")
    o1, o2, o3 = T / "Task_1/out", T / "Task_2/out", T / "Task_3/out"
    sets = {
        f"seed{s} verb": ({"T1": t1(o1 / f"seed{s}_{MODEL}_verb_{way}.json", cats),
                           "T2": t2(o2 / f"seed{s}_{MODEL}_verb_{way}_calibration.json"),
                           "T3": t3(o3 / f"seed{s}_{MODEL}_verb_synonyms.json", cats)}, "verb")
        for s in seeds}
    if MODEL == "gemma":
      sets.update({
        "no-hinge verb": ({"T1": t1(o1 / f"nohinge_gemma_verb_{way}.json", cats),
                           "T2": t2(o2 / f"nohinge_gemma_verb_{way}_calibration.json"),
                           "T3": t3(o3 / "nohinge_gemma_verb_synonyms.json", cats)}, "verb"),
    })
    best = {"T1": t1(o1 / f"fsvec{way}_{MODEL}.json", cats),
            "T2": t2(o2 / (f"neo_calibration_{MODEL}.json" if way == 3 else f"bin2_calibration_{MODEL}.json")),
            "T3": t3(o3 / f"synonyms_{MODEL}.json", cats)}
    sets["best verb"] = (best, "verb")
    sets["best unbiased"] = (best, "unbiased")
    return cats, sets


def hard(v, cats):
    return cats[int(np.argmax(v))] if isinstance(v, list) else v


def report(way, seeds):
    cats, sets = load(way, seeds)
    cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11) if way == 3 or p != "a"]
    print(f"\n===== {MODEL}, {'three' if way == 3 else 'two'}-way, {len(cons)} concepts =====")
    print("predicted " + "/".join(cats) + " counts, and accuracy against the concept's POS")
    print(f"{'':16}" + "".join(f"{t:>20}" for t in ("T1", "T2", "T3")))
    for name, (src, tp) in sets.items():
        row = f"{name:16}"
        for t in ("T1", "T2", "T3"):
            h = [hard(src[t][f"en_{x}_{tp}"], cats) for x in cons]
            c = collections.Counter(h)
            acc = np.mean([p == POS[x[0]] for p, x in zip(h, cons)])
            row += f"{'/'.join(str(c[k]) for k in cats) + f'  acc {acc:.2f}':>20}"
        print(row)
    if way == 2:
        print("contrast (noun concepts called noun - verb concepts called noun), pt")
        for name, (src, tp) in sets.items():
            row = f"{name:16}"
            for t in ("T1", "T2", "T3"):
                h = {x: hard(src[t][f"en_{x}_{tp}"], cats) for x in cons}
                n = np.mean([h[x] == "noun" for x in cons if x[0] == "n"])
                v = np.mean([h[x] == "noun" for x in cons if x[0] == "v"])
                row += f"{100 * (n - v):>+20.1f}"
            print(row)
    vi = cats.index("verb")
    print("mean P(verb) (T1, T3), and paired differences")
    for t in ("T1", "T3"):
        m = {name: np.array([src[t][f"en_{x}_{tp}"][vi] for x in cons]) for name, (src, tp) in sets.items()}
        print(f"  {t}: " + "  ".join(f"{k} {v.mean():.3f}" for k, v in m.items()))
        pairs = [(f"seed{s} verb", ref, sub) for s in seeds
                 for ref, sub in (("best unbiased", None), ("best verb", None), ("best verb", "seed42"))]
        for a, b, sub in pairs + [("best verb", "best unbiased", None)]:
            keep = [i for i, x in enumerate(cons) if sub is None or x not in NOT_SEED42]
            if sub and len(keep) == len(cons):
                continue
            d = m[a][keep] - m[b][keep]
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            print(f"     {a} - {b}{' (seed-42 1-epoch only)' if sub else ''}: {d.mean():+.3f} "
                  f"(Wilcoxon p={p:.3f}, n={len(d)})")
    print("agreement of hard calls between verb-template runs (same concept)")
    runs = [f"seed{s} verb" for s in seeds] + [k for k in ("best verb", "no-hinge verb") if k in sets]
    for a, b in itertools.combinations(runs, 2):
        A, B = sets[a][0], sets[b][0]
        print(f"  {a:>14} vs {b:<14}" + "  ".join(
            f"{t} {np.mean([hard(A[t][f'en_{x}_verb'], cats) == hard(B[t][f'en_{x}_verb'], cats) for x in cons]):.2f}"
            for t in ("T1", "T2", "T3")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma", choices=("gemma", "qwen", "aya"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[48])
    args = ap.parse_args()
    MODEL = args.model
    NOT_SEED42 = not_seed42(MODEL)
    print(f"{MODEL}: vectors_best verb vectors that are not seed-42 1-epoch: {sorted(NOT_SEED42)}")
    for way in (3, 2):
        report(way, args.seeds)
