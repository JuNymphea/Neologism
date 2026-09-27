#!/usr/bin/env python
"""Read a part of speech off the synonyms the model gives for each vector.

Generated synonyms (synonyms_<m>.json): each item is lowercased and given a *soft*
part-of-speech distribution. Out of context many English words are ambiguous
(deposit, place, answer), so a single tag would bake a tagger's guess into the
result. A single word gets WordNet's sense-tagged frequency (SemCor lemma counts)
per category, falling back to its number of senses when it has no counts; a
multi-word item gets spaCy's tag of its root. The vector's distribution is the
mean over its (up to) 5 samples x 5 synonyms.

Scored candidates (synonym_scores_<m>.npz): each candidate's lowercase and
capitalised forms are combined, and the per-candidate baseline is its mean over
the random control vectors -- the "likely synonym for anything" prior, measured
in the same prompt rather than with a content-free stand-in. For an injected
real word the word itself is removed from the candidates, so the control cannot
be answered by copying.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

CATS = ("noun", "verb", "adj")
SHORT = {"n": "noun", "v": "verb", "a": "adj"}


def parse(text: str) -> list[str]:
    """Up to five synonyms from either format the models use."""
    head = text.strip()
    lines = [l for l in text.splitlines() if re.match(r"^\s*\d+[.)]\s*\S", l)]
    if lines:
        items = [re.sub(r"^\s*\d+[.)]\s*", "", l) for l in lines]
    else:
        head = re.split(r"\n\s*\n", head)[0]
        items = re.split(r"[”\"“]\s*[,;]?\s*(?:and|or)?\s*[“\"]?|,", head)
    out = []
    for it in items:
        it = re.sub(r"\(.*?\)", "", it)
        it = re.sub(r"\s[-–—:].*$", "", it)
        w = it.strip().strip("*“”\"'.,:;!- ").strip().lower()
        w = re.sub(r"^(and|or)\s+", "", w)
        if w and len(w.split()) <= 3 and re.search(r"[a-z]", w):
            out.append(w)
    return out[:5]


class Tagger:
    def __init__(self):
        from nltk.corpus import wordnet as wn
        import spacy
        self.wn = wn
        self.nlp = spacy.load("en_core_web_sm")
        self.cache: dict = {}

    def __call__(self, w: str):
        if w in self.cache:
            return self.cache[w]
        dist = None
        if " " not in w:
            cnt = collections.Counter()
            senses = collections.Counter()
            for pos, cat in (("n", "noun"), ("v", "verb"), ("a", "adj"), ("s", "adj")):
                base = self.wn.morphy(w, pos) or w
                for syn in self.wn.synsets(base, pos=pos):
                    senses[cat] += 1
                    for lem in syn.lemmas():
                        if lem.name().lower() == base:
                            cnt[cat] += lem.count()
            src = cnt if sum(cnt.values()) else senses
            if sum(src.values()):
                tot = sum(src[c] for c in CATS)
                dist = {c: src[c] / tot for c in CATS} if tot else None
        if dist is None:
            doc = self.nlp(w)
            root = [t for t in doc if t.dep_ == "ROOT"] or list(doc)
            p = {"NOUN": "noun", "PROPN": "noun", "VERB": "verb", "ADJ": "adj"}.get(root[0].pos_)
            dist = {c: float(c == p) for c in CATS} if p else None
        self.cache[w] = dist
        return dist


def group(label: str):
    if label.startswith("real_"):
        return "real", SHORT[label.split("_")[1]]
    if label.startswith("rand"):
        return "random", None
    return "trained", SHORT[label.split("_")[1]]


def report(name, P, labels, pos_keys=CATS):
    """P: label -> {cat: prob}. Prints controls, then trained-vector results."""
    from scipy.stats import fisher_exact
    pred = {k: max(pos_keys, key=lambda c: P[k][c]) for k in P}
    print(f"\n==================== {name} ====================")
    real = [k for k in P if group(k)[0] == "real" and group(k)[1] in pos_keys]
    if real:
        acc = np.mean([pred[k] == group(k)[1] for k in real])
        per = {c: np.mean([pred[k] == c for k in real if group(k)[1] == c]) for c in pos_keys}
        print(f"  注入真实词  准确率 {acc:.3f}  (n={len(real)})  "
              + "  ".join(f"{c}:{v:.2f}" for c, v in per.items()))
    rnd = [k for k in P if group(k)[0] == "random"]
    if rnd:
        c = collections.Counter(pred[k] for k in rnd)
        print(f"  随机向量    预测分布 " + " / ".join(f"{x}:{c[x]}" for x in pos_keys)
              + "   平均 P " + " ".join(f"{x}:{np.mean([P[k][x] for k in rnd]):.2f}" for x in pos_keys))
    tr = [k for k in P if group(k)[0] == "trained" and group(k)[1] in pos_keys
          and (len(pos_keys) == 3 or not k.endswith("_adj"))]
    for lang in ("en", "zh", "all"):
        ks = [k for k in tr if lang == "all" or k.startswith(lang + "_")]
        hit = np.mean([pred[k] == group(k)[1] for k in ks])
        c = collections.Counter(pred[k] for k in ks)
        print(f"  训练向量[{lang}] 匹配 {hit:.3f} (n={len(ks)}, 随机 {1/len(pos_keys):.3f})  "
              "预测 " + " / ".join(f"{x}:{c[x]}" for x in pos_keys))
    # contrast: P(noun) for noun-concepts vs verb-concepts, concept-level means too
    nn = [P[k]["noun"] for k in tr if group(k)[1] == "noun"]
    vv = [P[k]["noun"] for k in tr if group(k)[1] == "verb"]
    t = [[sum(pred[k] == "noun" for k in tr if group(k)[1] == "noun"),
          sum(pred[k] != "noun" for k in tr if group(k)[1] == "noun")],
         [sum(pred[k] == "noun" for k in tr if group(k)[1] == "verb"),
          sum(pred[k] != "noun" for k in tr if group(k)[1] == "verb")]]
    _, p = fisher_exact(t)
    print(f"  对比量: 概念名词判 noun {t[0][0]}/{sum(t[0])}={t[0][0]/sum(t[0]):.3f}  "
          f"概念动词判 noun {t[1][0]}/{sum(t[1])}={t[1][0]/sum(t[1]):.3f}  "
          f"差 {100*(t[0][0]/sum(t[0])-t[1][0]/sum(t[1])):.1f}pt  p={p:.2e}   "
          f"| 平均P(noun) {np.mean(nn):.3f} vs {np.mean(vv):.3f}")
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--models", default="gemma,qwen,aya")
    args = ap.parse_args()
    tag = Tagger()
    summary = {}
    for m in args.models.split(","):
        # ---- generated synonyms
        R = json.loads((args.dir / f"synonyms_{m}.json").read_text())["results"]
        P, n_items, n_tagged, n_empty = {}, 0, 0, 0
        for k, samples in R.items():
            acc = np.zeros(3); w = 0
            for s in samples:
                syn = parse(s["raw"])
                if not syn:
                    n_empty += 1
                for x in syn:
                    n_items += 1
                    d = tag(x)
                    if d:
                        n_tagged += 1
                        acc += [d[c] for c in CATS]; w += 1
            P[k] = {c: (acc[i] / w if w else 1 / 3) for i, c in enumerate(CATS)}
        print(f"\n######## {m}: {len(R)} 向量, {n_items} 个近义词, 可判词性 {n_tagged} "
              f"({100*n_tagged/max(n_items,1):.1f}%), 解析为空的样本 {n_empty}")
        gen3 = report(f"{m} 生成近义词 三分类", P, list(P))
        P2 = {k: {"noun": v["noun"] / (v["noun"] + v["verb"] + 1e-12),
                  "verb": v["verb"] / (v["noun"] + v["verb"] + 1e-12), "adj": 0} for k, v in P.items()}
        gen2 = report(f"{m} 生成近义词 二分类", P2, list(P2), pos_keys=("noun", "verb"))

        # ---- scored candidates
        z = np.load(args.dir / f"synonym_scores_{m}.npz")
        L, labels = z["logp"], list(z["labels"])
        fc, cands, cpos = z["form_cand"], z["cands"], z["cand_pos"]
        C = np.full((L.shape[0], len(cands)), -np.inf)
        for f, c in enumerate(fc):
            C[:, c] = np.logaddexp(C[:, c], L[:, f])
        rnd = [i for i, k in enumerate(labels) if k.startswith("rand")]
        base = C[rnd].mean(0)
        for tagname, S in (("原始", C), ("减随机基线", C - base)):
            for pk in (CATS, ("noun", "verb")):
                Pb = {}
                for i, k in enumerate(labels):
                    s = S[i].copy()
                    if k.startswith("real_"):
                        s[cands == k.split("_", 2)[2]] = -np.inf
                    s = s - s.max()
                    e = np.exp(s)
                    mass = {c: e[cpos == c].sum() for c in pk}
                    tot = sum(mass.values())
                    Pb[k] = {c: (mass.get(c, 0) / tot) for c in CATS}
                report(f"{m} 候选打分 {tagname} {'三' if len(pk)==3 else '二'}分类", Pb, labels, pos_keys=pk)
        summary[m] = {"gen3": gen3, "gen2": gen2}
    (args.dir / "synonym_predictions.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
