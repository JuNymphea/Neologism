#!/usr/bin/env python
"""Chinese Task 1 and Task 3 on the Chinese vectors, with real-word checks.

Task 1 (run_task1.py --lang zh): 词语“X”的词性是 with 名词/动词/形容词, three shots
(桌子/喝/高兴) averaged over their orders; binary with 桌子/喝.
Task 3 (synonyms.py --lang zh): five Chinese synonyms per sample, five samples;
each synonym tagged by jieba's dictionary (ICTCLAS tags distinguish adjectives,
which UD Chinese mostly tags VERB):
    n* -> noun   v* -> verb   a, ad, ag, z, b -> adj
    vn -> noun/verb half and half,  an -> adj/noun half and half
    anything else (idioms, proper-name misses) -> not counted
A word jieba's dictionary lacks is segmented: verb-initial -> verb, otherwise
the last segment's tag.

Real words: the Task 2 zh control words (UD dominance >= 0.9 and matching jieba
tag, 81 per category) -- as text for Task 1, as embedding rows injected into the
new token for Task 3.

    python scripts/Tasks/controls/analyze_zh_t13.py [--models gemma qwen aya]
"""
import argparse, collections, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, fisher_exact

T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "Task_3"))
CATS = ("noun", "verb", "adj")
POS = {"n": "noun", "v": "verb", "a": "adj"}
TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")


class TaggerZH:
    def __init__(self):
        import jieba, jieba.posseg as ps
        jieba.setLogLevel(60)
        self.ps, self.tab = ps, ps.dt.word_tag_tab
        self.cache = {}

    @staticmethod
    def dist(tag):
        if tag is None:
            return None
        if tag == "vn":
            return {"noun": .5, "verb": .5, "adj": 0.}
        if tag == "an":
            return {"noun": .5, "verb": 0., "adj": .5}
        if tag.startswith("n"):
            return {"noun": 1., "verb": 0., "adj": 0.}
        if tag.startswith("v"):
            return {"noun": 0., "verb": 1., "adj": 0.}
        if tag in ("a", "ad", "ag", "z", "b"):
            return {"noun": 0., "verb": 0., "adj": 1.}
        return None

    def __call__(self, w):
        if w in self.cache:
            return self.cache[w]
        tag = self.tab.get(w)
        if tag is None:
            segs = self.ps.lcut(w)
            if segs:
                tag = segs[0].flag if (len(segs) > 1 and segs[0].flag.startswith("v")) else segs[-1].flag
        self.cache[w] = self.dist(tag)
        return self.cache[w]


tag = None
_en = None


def en_items(raw):
    """English synonyms in a reply to the Chinese prompt (some vectors answer in
    English); they name a part of speech just as well, read with WordNet."""
    global _en
    if _en is None:
        from analyze_synonyms import parse, Tagger
        _en = (parse, Tagger())
    return [w for w in _en[0](raw) if re.fullmatch(r"[a-z][a-z '\-]*", w)]


def t3(path, cats):
    """Mean POS distribution over a vector's tagged synonyms, or None when not
    one synonym could be tagged -- a missing reading, not a uniform one (a
    uniform vector would fall to 'noun' at the argmax)."""
    out = {}
    for k, samples in json.load(open(path))["results"].items():
        acc, n = np.zeros(3), 0
        for s in samples:
            items = [(w, tag) for w in s["synonyms"]]
            if not items:
                en_items(s["raw"])
                items = [(w, _en[1]) for w in en_items(s["raw"])]
            for w, tg in items:
                d = tg(w)
                if d:
                    acc += [d[c] for c in CATS]; n += 1
        if not n:
            out[k] = None
            continue
        p = np.array([(acc / n)[CATS.index(c)] for c in cats])
        out[k] = list(p / p.sum()) if p.sum() else None
    return out


def t1(path, cats):
    d = json.load(open(path))["probabilities"]
    d = d.get("neologism", d)
    return {k: [v[c] for c in cats] for k, v in d.items()}


def hard(v, cats):
    return None if v is None else cats[int(np.argmax(v))]


def main():
    global tag
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gemma", "qwen", "aya"])
    args = ap.parse_args()
    tag = TaggerZH()
    o1, o3 = T / "Task_1/out", T / "Task_3/out"
    for m in args.models:
        print(f"\n################ {m} (zh) ################")
        # ---- real words ----
        for way in (3, 2):
            r = json.load(open(o1 / f"t1zh{way}_real_{m}.json"))["accuracy"]
            print(f"T1 real words {way}-way: overall {r['overall']:.3f}  " +
                  "  ".join(f"{p} {r[p]['accuracy']:.3f}" for p in (CATS if way == 3 else CATS[:2])))
        syn = t3(o3 / f"t3zh_synonyms_{m}.json", CATS)
        real = {k: v for k, v in syn.items() if k.startswith("real_")}
        by = collections.Counter(); hit = collections.Counter()
        for k, v in real.items():
            g = POS[k.split("_")[1]]; by[g] += 1; hit[g] += hard(v, CATS) == g   # missing counts as wrong
        print(f"T3 real words (rows injected) 3-way: overall {sum(hit.values()) / max(1, sum(by.values())):.3f}  " +
              "  ".join(f"{p} {hit[p]}/{by[p]}" for p in CATS))
        nv = {k: v for k, v in real.items() if k.split("_")[1] in "nv" and v is not None}
        h2 = sum((v[0] > v[1]) == (k.split("_")[1] == "n") for k, v in nv.items())
        print(f"T3 real words 2-way: {h2 / max(1, len(nv)):.3f} (n={len(nv)})")
        # ---- vectors ----
        for way in (3, 2):
            cats = CATS if way == 3 else ("noun", "verb")
            cons = [f"{p}_{i}" for p in ("a", "n", "v") for i in range(1, 11) if way == 3 or p != "a"]
            D = {"T1": t1(o1 / f"t1zh{way}_vec_{m}.json", cats), "T3": t3(o3 / f"t3zh_synonyms_{m}.json", cats)}
            print(f"--- vectors, {way}-way ({len(cons)} concepts): calls "
                  + "/".join(cats) + " per concept POS, accuracy")
            for tp in TEMPLATES:
                line = f"  {tp:9}"
                for task in ("T1", "T3"):
                    h = {x: hard(D[task][f"zh_{x}_{tp}"], cats) for x in cons}
                    cell = " ".join("/".join(str(sum(h[x] == c for x in cons if POS[x[0]] == cp)) for c in cats)
                                    for cp in cats)
                    ok = [x for x in cons if h[x] is not None]
                    acc = np.mean([h[x] == POS[x[0]] for x in ok]) if ok else float("nan")
                    miss = len(cons) - len(ok)
                    extra = ""
                    if way == 2:
                        a = sum(h[x] == "noun" for x in cons if x[0] == "n")
                        b = sum(h[x] == "noun" for x in cons if x[0] == "v")
                        extra = f" contrast {10 * (a - b):+d}pt (Fisher p={fisher_exact([[a, 10 - a], [b, 10 - b]])[1]:.2f})"
                    line += f" | {task} {cell} acc {acc:.2f}{extra}" + (f" [missing {miss}]" if miss else "")
                print(line)
            print("  template effect: P(template POS) under template - under unbiased, paired")
            for tp in ("noun", "verb", "adj"):
                if tp not in cats:
                    continue
                k = cats.index(tp)
                parts = []
                for task in ("T1", "T3"):
                    d = np.array([D[task][f"zh_{x}_{tp}"][k] - D[task][f"zh_{x}_unbiased"][k] for x in cons
                                  if D[task][f"zh_{x}_{tp}"] is not None and D[task][f"zh_{x}_unbiased"] is not None])
                    p = wilcoxon(d).pvalue if np.any(d) else 1.0
                    parts.append(f"{task} {d.mean():+.3f} (p={p:.3f})")
                print(f"    {tp:5}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
