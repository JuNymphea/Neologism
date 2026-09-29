#!/usr/bin/env python
"""Chinese Task 3 read as frame compatibility (Task_3/llm_frames_zh.py).

For a vector, C = (N, V, A) is the share of its synonyms (valid items, all five
samples pooled) that fit the noun, verb and adjective frame; a synonym may fit
several, so the three need not sum to 1. Reported:

  real words   the 3x3 table of mean C by the word's category, the share of
               words whose own frame is the (possibly tied) highest, and the
               specificity C[own] - mean C[other two]. Words the LLM finds do
               not fit their own frame, or are phrases (很大, 模特 ...), are
               dropped from the list first, and listed.
  vectors      per template, mean C by concept category; under unbiased, the
               specificity by concept category (Wilcoxon against 0); and the
               template effect C[template frame] under the template minus under
               unbiased, paired by concept (Wilcoxon).

    python scripts/Tasks/controls/analyze_zh_frames.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import analyze_zh_t13 as Z  # noqa: E402

T = HERE.parents[1]
F = ("N", "V", "A")
CAT = {"n": 0, "v": 1, "a": 2}
NAME = {"n": "noun", "v": "verb", "a": "adj"}
TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")
FRAME_OF = {"noun": 0, "verb": 1, "adj": 2}


def load_samples(path):
    """A Task 3 output with its tie-break extras (<stem>_extra<r>.json) pooled in:
    vectors whose frames tied got five more samples per round, until they did not."""
    path = Path(path)
    res = {k: list(v) for k, v in json.load(open(path))["results"].items()}
    for extra in sorted(path.parent.glob(f"{path.stem}_extra*.json")):
        for k, v in json.load(open(extra))["results"].items():
            res.setdefault(k, []).extend(v)
    return res


def compat(path, cache):
    out = {}
    for k, samples in load_samples(path).items():
        rows = []
        for s in samples:
            for w in (s["synonyms"] or Z.en_items(s["raw"])):
                c = cache.get(w.strip())
                if c and c["valid"]:
                    rows.append([c[f] for f in F])
        out[k] = np.mean(rows, axis=0) if rows else None
    return out


def spec(c, i):
    return c[i] - np.mean([c[j] for j in range(3) if j != i])


def main():
    cache = json.load(open(T / "Task_3/out/zh_frames_llm.json"))
    W = json.load(open(T / "controls/zh_words3.json"))["words"]
    keep = {p: [w for w in ws if cache[w][F[FRAME_OF[p]]] and cache[w]["single"] and cache[w]["valid"]]
            for p, ws in W.items()}
    dropped = {p: [w for w in W[p] if w not in keep[p]] for p in W}
    print("real-word list: kept " + ", ".join(f"{p} {len(keep[p])}/{len(W[p])}" for p in W))
    print("  dropped: " + "; ".join(f"{p}: {' '.join(v)}" for p, v in dropped.items() if v))
    files = {m: T / f"Task_3/out/t3zh_synonyms_{m}.json" for m in ("gemma", "qwen", "aya")}
    for m, f in files.items():
        C = compat(f, cache)
        print(f"\n################ {m}: real words (rows injected), current-code run ################")
        print("  mean compatibility N / V / A by the word's category; own-frame-highest; specificity")
        for p in ("noun", "verb", "adj"):
            cs = [C[f"real_{p[0]}_{w}"] for w in keep[p] if C.get(f"real_{p[0]}_{w}") is not None]
            i = FRAME_OF[p]
            top = np.mean([c[i] >= max(c) - 1e-9 for c in cs])
            sp = [spec(c, i) for c in cs]
            print(f"  {p:5} n={len(cs):3}  " + " / ".join(f"{np.mean([c[j] for c in cs]):.2f}" for j in range(3))
                  + f"   own highest {top:.2f}   specificity {np.mean(sp):+.2f}")
    sets = [(f"{m}, current code (vectors_best zh)", T / f"Task_3/out/t3zh_synonyms_{m}.json") for m in ("gemma", "qwen", "aya")]
    sets += [(f"{m}, original code (fn30 zh)", T / f"Task_3/out/legacyzh_t3zh_synonyms_{m}.json")
             for m in ("gemma", "qwen", "aya") if (T / f"Task_3/out/legacyzh_t3zh_synonyms_{m}.json").exists()]
    for name, f in sets:
        C = compat(f, cache)
        cons = [f"{p}_{i}" for p in "anv" for i in range(1, 11)]
        miss = sum(C.get(f"zh_{x}_{t}") is None for x in cons for t in TEMPLATES)
        print(f"\n================ vectors: {name}  (missing {miss}/150) ================")
        print("  mean compatibility N / V / A for noun | verb | adj concepts")
        for t in TEMPLATES:
            cells = []
            for p in "nva":
                cs = [C[f"zh_{p}_{i}_{t}"] for i in range(1, 11) if C.get(f"zh_{p}_{i}_{t}") is not None]
                cells.append("/".join(f"{np.mean([c[j] for c in cs]):.2f}" for j in range(3)))
            print(f"  {t:9} " + " | ".join(cells))
        print("  unbiased specificity C[own] - mean C[other two], by concept category (Wilcoxon vs 0)")
        for p in "nva":
            sp = [spec(C[f"zh_{p}_{i}_unbiased"], CAT[p]) for i in range(1, 11) if C.get(f"zh_{p}_{i}_unbiased") is not None]
            pv = wilcoxon(sp).pvalue if np.any(sp) else 1.0
            print(f"    {NAME[p]:5} concepts: {np.mean(sp):+.3f} (p={pv:.3f}, n={len(sp)})")
        nv = [spec(C[f"zh_{p}_{i}_unbiased"], CAT[p]) for p in "nv" for i in range(1, 11) if C.get(f"zh_{p}_{i}_unbiased") is not None]
        print(f"    noun+verb concepts pooled: {np.mean(nv):+.3f} (p={wilcoxon(nv).pvalue:.3f}, n={len(nv)})")
        print("  template effect: C[template frame] under template - under unbiased (paired, 30 concepts)")
        for t in ("noun", "verb", "adj"):
            j = FRAME_OF[t]
            d = [C[f"zh_{x}_{t}"][j] - C[f"zh_{x}_unbiased"][j] for x in cons
                 if C.get(f"zh_{x}_{t}") is not None and C.get(f"zh_{x}_unbiased") is not None]
            pv = wilcoxon(d).pvalue if np.any(d) else 1.0
            print(f"    {t:5}: {np.mean(d):+.3f} (p={pv:.3f}, n={len(d)})")


if __name__ == "__main__":
    main()
