#!/usr/bin/env python
"""gemma en, verb template: no-hinge vectors against their hinge counterparts.

Same concept, same seed-42 init, same data; only lambda_h differs (0 vs 0.1). The
question is whether dropping the hinge makes the verb template read as verb.
Reference: the hinge vectors of the same concept under the unbiased template.
"""
import json, sys, collections
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, fisher_exact
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "Task_3"))
from analyze_synonyms import parse, Tagger, CATS  # noqa
tag = Tagger()
POS = {"n": "noun", "v": "verb", "a": "adj"}

def t1(path): return {k: [v[c] for c in CATS] for k, v in json.load(open(path))["probabilities"]["neologism"].items()}
def t3(path):
    out = {}
    for k, samples in json.load(open(path))["results"].items():
        acc = np.zeros(3); w = 0
        for s in samples:
            for x in parse(s["raw"]):
                d = tag(x)
                if d: acc += [d[c] for c in CATS]; w += 1
        out[k] = list(acc / w) if w else [1 / 3] * 3
    return out
syn = json.load(open(T / "Task_3/out/synonym_predictions.json"))
H = {"T1": t1(T / "Task_1/out/fsvec3_gemma.json"),
     "T2": json.load(open(T / "Task_2/out/neo_calibration_gemma.json"))["neologisms"],
     "T3": t3(T / "Task_3/out/synonyms_gemma.json")}
N = {"T1": t1(T / "Task_1/out/nohinge_gemma_verb_3.json"),
     "T2": json.load(open(T / "Task_2/out/nohinge_gemma_verb_3_calibration.json"))["neologisms"],
     "T3": t3(T / "Task_3/out/nohinge_gemma_verb_synonyms.json")}
cons = sorted({k.split("_")[1] + "_" + k.split("_")[2] for k in N["T1"]})
hard = lambda v: CATS[int(np.argmax(v))] if isinstance(v, list) else v
print(f"{'':22}{'T1 n/v/a':>12}{'T2 n/v/a':>12}{'T3 n/v/a':>12}")
for name, src, tp in (("hinge unbiased", H, "unbiased"), ("hinge verb", H, "verb"), ("no-hinge verb", N, "verb")):
    row = f"{name:22}"
    for t in ("T1", "T2", "T3"):
        c = collections.Counter(hard(src[t][f"en_{x}_{tp}"]) for x in cons)
        row += f"{'%d/%d/%d' % (c['noun'], c['verb'], c['adj']):>12}"
    print(row)
for t in ("T1", "T3"):
    for ref, rsrc, rtp in (("hinge verb", H, "verb"), ("hinge unbiased", H, "unbiased")):
        d = [N[t][f"en_{x}_verb"][1] - rsrc[t][f"en_{x}_{rtp}"][1] for x in cons]
        print(f"  {t} P(verb): no-hinge verb − {ref} = {np.mean(d):+.3f} (Wilcoxon p={wilcoxon(d).pvalue:.3f}, n={len(d)})")
for t in ("T1", "T2", "T3"):
    m = lambda src, tp: np.mean([hard(src[t][f"en_{x}_{tp}"]) == POS[x[0]] for x in cons])
    print(f"  {t} concept-POS match: hinge verb {m(H, 'verb'):.2f}  no-hinge verb {m(N, 'verb'):.2f}")
