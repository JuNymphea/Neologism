#!/usr/bin/env python
"""Do the unbiased vectors carry their concept's part of speech? Concept-level tests.

Six settings: gemma / qwen / aya x English / Chinese, original training code on
fn30, seed 42, 1 epoch, unbiased template. Each concept gets one score from its
three task calls (T1, T2, T3) -- the calls on one concept are not independent,
so the concept, not the call, is the unit.

  three-way   statistic: accuracy = mean over the 30 concepts of the share of
              the three tasks that call the concept's own category.
  two-way     statistic: contrast = mean share of tasks calling noun for the 10
              noun concepts minus the same for the 10 verb concepts.

Null: the concepts' category labels are permuted (10/10/10, or 10/10), the
calls kept -- so a model that calls everything noun has the chance level its
own calls imply, not 1/3. p is one-sided (statistic >= observed), 20,000
permutations. CI: bootstrap over concepts, stratified by category. Holm across
the six settings; the pooled test sums the six statistics under independent
permutations.

    python scripts/Tasks/controls/unbiased_permutation.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyze_seeds import t1 as t1en, t2 as t2en, t3 as t3en  # noqa: E402
import analyze_zh_t13 as Z  # noqa: E402
import analyze_zh_spec as ZS  # noqa: E402

T = HERE.parents[1]
CATS = ("noun", "verb", "adj")
POS = {"n": "noun", "v": "verb", "a": "adj"}
N_PERM, N_BOOT = 20000, 10000
rng = np.random.default_rng(0)


def argmax_call(v, cats):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return cats[int(np.argmax(v))]


def calls(lang, m, way):
    cats = CATS if way == 3 else ("noun", "verb")
    cons = [f"{p}_{i}" for p in ("n", "v", "a") for i in range(1, 11) if way == 3 or p != "a"]
    if lang == "en":
        D = [t1en(T / f"Task_1/out/legacy_{m}_unbiased_{way}.json", cats),
             t2en(T / f"Task_2/out/legacy_{m}_unbiased_{way}_calibration.json"),
             t3en(T / f"Task_3/out/legacy_{m}_unbiased_synonyms.json", cats)]
        key = "en_{}_unbiased"
    else:
        Z.tag = Z.tag or Z.TaggerZH()
        D = [Z.t1(T / f"Task_1/out/legacyzh_t1zh{way}_{m}.json", cats),
             json.load(open(T / f"Task_2/out_zh/legacyzh_vec{way}_calibration_{m}.json"))["neologisms"],
             ZS.t3_frames(T / f"Task_3/out/legacyzh_t3zh_synonyms_{m}.json", cats)]
        key = "zh_{}_unbiased"
    labels = np.array([POS[c[0]] for c in cons])
    C = np.array([[argmax_call(d[key.format(c)], cats) for d in D] for c in cons])   # concepts x 3 tasks
    return labels, C


def acc_stat(labels, C):
    return np.mean((C == labels[:, None]).mean(1))


def contrast_stat(labels, C):
    noun_share = (C == "noun").mean(1)
    return noun_share[labels == "noun"].mean() - noun_share[labels == "verb"].mean()


def test(labels, C, stat):
    obs = stat(labels, C)
    null = np.array([stat(rng.permutation(labels), C) for _ in range(N_PERM)])
    p = (np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)
    boots = []
    groups = [np.where(labels == c)[0] for c in np.unique(labels)]
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        boots.append(stat(labels[idx], C[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, null.mean(), p, (lo, hi), null


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); running = 0
    for r, i in enumerate(order):
        running = max(running, min(1, (len(ps) - r) * ps[i]))
        out[i] = running
    return out


def main():
    settings = [(l, m) for l in ("en", "zh") for m in ("gemma", "qwen", "aya")]
    for way, stat, name in ((3, acc_stat, "three-way accuracy"), (2, contrast_stat, "two-way noun/verb contrast")):
        print(f"\n===== {name} (unbiased, concept-level permutation) =====")
        print(f"{'':10}{'observed':>9} {'null mean':>9} {'excess':>8} {'95% CI':>16} {'p':>8} {'p Holm':>8}")
        rows, nulls = [], []
        for l, m in settings:
            labels, C = calls(l, m, way)
            obs, nm, p, ci, null = test(labels, C, stat)
            rows.append((l, m, obs, nm, p, ci)); nulls.append(null)
        ph = holm(np.array([r[4] for r in rows]))
        for (l, m, obs, nm, p, ci), q in zip(rows, ph):
            print(f"{l.upper()} {m:6}{obs:9.3f} {nm:9.3f} {obs - nm:+8.3f}   [{ci[0]:.3f}, {ci[1]:.3f}] {p:8.4f} {q:8.4f}")
        tot = sum(r[2] for r in rows)
        null_sum = np.sum(nulls, axis=0)
        pp = (np.sum(null_sum >= tot - 1e-12) + 1) / (N_PERM + 1)
        print(f"pooled over six: sum of statistics {tot:.3f} vs null mean {null_sum.mean():.3f}, p = {pp:.5f}")


if __name__ == "__main__":
    main()
