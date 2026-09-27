#!/usr/bin/env python
"""Template effect before and after removing the random init (initswap variants)."""
import json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact, wilcoxon
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "Task_3"))
from analyze_synonyms import parse, Tagger, CATS  # noqa
tag = Tagger()
POS = {"n": "noun", "v": "verb", "a": "adj"}

def t1_soft(path):
    return {k: [v[c] for c in CATS] for k, v in json.load(open(path))["probabilities"]["neologism"].items()}

def t3_soft(path):
    out = {}
    for k, samples in json.load(open(path))["results"].items():
        acc = np.zeros(3); w = 0
        for s in samples:
            for x in parse(s["raw"]):
                d = tag(x)
                if d: acc += [d[c] for c in CATS]; w += 1
        out[k] = list(acc / w) if w else [1/3] * 3
    return out

def hard(P): return {k: CATS[int(np.argmax(v))] if isinstance(v, list) else v for k, v in P.items()}

def load(m, v):
    if v == "orig":
        syn = json.load(open(T / "Task_3/out/synonym_predictions.json"))
        return {"T1": {k: x for k, x in t1_soft(T / f"Task_1/out/fsvec3_{m}.json").items() if k.startswith("en_")},
                "T2": {k: x for k, x in json.load(open(T / f"Task_2/out/neo_calibration_{m}.json"))["neologisms"].items() if k.startswith("en_")},
                "T3": {k: x for k, x in t3_soft(T / f"Task_3/out/synonyms_{m}.json").items() if k.startswith("en_")}}
    o = f"initswap_{m}_{v}"
    return {"T1": t1_soft(T / f"Task_1/out/{o}_3.json"),
            "T2": json.load(open(T / f"Task_2/out/{o}_3_calibration.json"))["neologisms"],
            "T3": t3_soft(T / f"Task_3/out/{o}_synonyms.json")}

def report(m, v, P):
    H = {t: hard(P[t]) for t in P}
    tpls = ("unbiased", "mixed", "noun", "verb", "adj")
    print(f"\n  [{v}]  每个模版的预测 n/v/a（每格 30）")
    print("   " + "".join(f"{t:>14}" for t in ("T1", "T2", "T3")))
    for tp in tpls:
        row = f"   {tp:9}"
        for t in ("T1", "T2", "T3"):
            c = collections.Counter(H[t][k] for k in H[t] if k.endswith("_" + tp))
            row += f"{'%d/%d/%d' % (c['noun'], c['verb'], c['adj']):>14}"
        print(row)
    for t in ("T1", "T2", "T3"):
        parts = []
        for ci, tp in enumerate(("noun", "verb", "adj")):
            a = sum(H[t][k] == tp for k in H[t] if k.endswith("_" + tp))
            b = sum(H[t][k] == tp for k in H[t] if k.endswith("_unbiased"))
            _, p = fisher_exact([[a, 30 - a], [b, 30 - b]])
            s = f"{tp}:{a}vs{b}{'*' if p < 0.05 else ''}"
            if t != "T2":   # paired soft shift on the same concepts
                cons = sorted({k.split("_")[1] + "_" + k.split("_")[2] for k in P[t]})
                d = [P[t][f"en_{c}_{tp}"][ci] - P[t][f"en_{c}_unbiased"][ci] for c in cons]
                pw = wilcoxon(d).pvalue if any(d) else 1
                s += f"(Δ{np.mean(d):+.2f}{'*' if pw < 0.05 else ''})"
            parts.append(s)
        match = np.mean([H[t][k] == POS[k.split("_")[1]] for k in H[t]])
        print(f"   {t} 模版效应 " + "  ".join(parts) + f"   | 概念词性符合 {match:.2f}")

for m in ("gemma", "qwen", "aya"):
    print(f"\n################ {m} en ################")
    for v in ("orig", "mu", "muw", "delta"):
        try: report(m, v, load(m, v))
        except FileNotFoundError as e: print(f"  [{v}] missing: {e.filename}")
