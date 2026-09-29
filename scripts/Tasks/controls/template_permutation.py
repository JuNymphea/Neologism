#!/usr/bin/env python
"""Do the training templates move the part of speech, and toward their own? (three-way)

Six conditions (gemma / qwen / aya x English / Chinese; original training code,
fn30 data, seed 42, 1 epoch). Per condition: 30 concepts x 5 templates
(unbiased, noun, verb, adj, mixed) x 3 probes, each a three-way call. The concept
is the block: its five vectors share the training data and differ only in the
template, so under the null (templates have no effect) its template labels are
exchangeable, and every test permutes them within concept. The six conditions
share the 30 concepts, so the pooled test applies one within-concept shuffle to
all six at once. share(c, t, p) is the fraction of the three probes that call p
for concept c's template-t vector.

  1 omnibus      any dependence of the call on the template: a chi-square-type
                 statistic on the template x call counts (p by permutation of
                 all five labels within concept).
  2 congruence   D_t = share(c, t, t) - share(c, unbiased, t) for t in noun,
                 verb, adj: does a template raise its own category over the
                 unbiased vector? (per template: swap t and unbiased within
                 concept; the average of the three: permute all five labels.)
  3 specificity  among the three framed templates only: S = mean over concepts
                 of 1/3 * sum_t share(c, t, t), the diagonal of the template x
                 call table; null mean exactly 1/3. Beats a generic "any frame
                 shifts toward noun" effect, which moves all three templates
                 alike. Per template: share(c, t, t) - mean of share(c, t', t)
                 over the other two framed templates. Also per probe.
  4 mixed        does the mixed template differ from unbiased at all (total
                 variation distance between their call distributions; swap
                 within concept)?

p one-sided (>= observed), 20,000 permutations, Holm across the six conditions.

    python scripts/Tasks/controls/template_permutation.py
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyze_seeds import t1 as t1en, t2 as t2en, t3 as t3en  # noqa: E402
import analyze_zh_t13 as Z  # noqa: E402
import analyze_zh_spec as ZS  # noqa: E402

T = HERE.parents[1]
CATS = ("noun", "verb", "adj")
TPL = ("unbiased", "noun", "verb", "adj", "mixed")
CONS = [f"{p}_{i}" for p in ("n", "v", "a") for i in range(1, 11)]
N_PERM = 20000
rng = np.random.default_rng(0)


def call(v):
    if v is None:
        return -1
    if isinstance(v, str):
        return CATS.index(v)
    return int(np.argmax(v))


def load(lang, m):
    """X[concept, template, probe] = index of the called category."""
    X = np.full((30, 5, 3), -1)
    for j, t in enumerate(TPL):
        if lang == "en":
            D = [t1en(T / f"Task_1/out/legacy_{m}_{t}_3.json", CATS),
                 t2en(T / f"Task_2/out/legacy_{m}_{t}_3_calibration.json"),
                 t3en(T / f"Task_3/out/legacy_{m}_{t}_synonyms.json", CATS)]
            key = "en_{}_" + t
        else:
            Z.tag = Z.tag or Z.TaggerZH()
            if j == 0:
                load.zh = [Z.t1(T / f"Task_1/out/legacyzh_t1zh3_{m}.json", CATS),
                           json.load(open(T / f"Task_2/out_zh/legacyzh_vec3_calibration_{m}.json"))["neologisms"],
                           ZS.t3_frames(T / f"Task_3/out/legacyzh_t3zh_synonyms_{m}.json", CATS)]
            D, key = load.zh, "zh_{}_" + t
        for i, c in enumerate(CONS):
            X[i, j] = [call(d[key.format(c)]) for d in D]
    assert (X >= 0).all(), f"{lang} {m}: missing calls"
    return X


def share(X):
    """S[concept, template, category]"""
    return np.stack([(X == k).mean(-1) for k in range(3)], -1)


def omnibus(S):
    n = S.sum(0)                                  # template x category (in concept units)
    e = n.sum(1, keepdims=True) * n.sum(0, keepdims=True) / n.sum()
    return float(((n - e) ** 2 / np.where(e > 0, e, 1)).sum())


def congruence(S):
    return np.mean([(S[:, TPL.index(t), k] - S[:, 0, k]).mean() for k, t in enumerate(CATS)])


def cong_t(S, k):
    t = TPL.index(CATS[k])
    return (S[:, t, k] - S[:, 0, k]).mean()


FR = [TPL.index(t) for t in CATS]                 # the three framed templates, in category order


def specificity(S):
    return np.mean([S[:, FR[k], k] for k in range(3)])


def spec_t(S, k):
    others = [FR[j] for j in range(3) if j != k]
    return (S[:, FR[k], k] - S[:, others, k].mean(1)).mean()


def mixed_tv(S):
    return 0.5 * np.abs(S[:, 4].mean(0) - S[:, 0].mean(0)).sum()


def perm_within(S, cols, perms):
    """Apply one permutation of the template columns `cols` per concept."""
    P = S.copy()
    for i, pi in enumerate(perms):
        P[i, cols] = S[i, [cols[x] for x in pi]]
    return P


def run(Ss, stat, cols, n=N_PERM):
    """Observed per condition, per-condition p, pooled (sum) p under a shared shuffle."""
    obs = np.array([stat(S) for S in Ss])
    ge = np.zeros(len(Ss)); ge_sum = 0; tot = obs.sum()
    for _ in range(n):
        perms = [rng.permutation(len(cols)) for _ in range(30)]
        vals = np.array([stat(perm_within(S, cols, perms)) for S in Ss])
        ge += vals >= obs - 1e-12
        ge_sum += vals.sum() >= tot - 1e-12
    return obs, (ge + 1) / (n + 1), (ge_sum + 1) / (n + 1)


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); run_ = 0
    for r, i in enumerate(order):
        run_ = max(run_, min(1, (len(ps) - r) * ps[i])); out[i] = run_
    return out


def main():
    conds = [(l, m) for l in ("en", "zh") for m in ("gemma", "qwen", "aya")]
    names = [f"{l.upper()} {m}" for l, m in conds]
    Xs = [load(l, m) for l, m in conds]
    Ss = [share(X) for X in Xs]

    def report(title, stat, cols, fmt="{:+.3f}", n=N_PERM):
        obs, p, pp = run(Ss, stat, cols, n)
        ph = holm(p)
        print(f"\n--- {title}")
        for nm, o, a, b in zip(names, obs, p, ph):
            print(f"  {nm:10} {fmt.format(o):>8}   p={a:.4f}  Holm={b:.4f}" + ("  (at the permutation floor)" if a <= 1.5 / (n + 1) else ""))
        print(f"  pooled     {fmt.format(obs.mean()):>8}   p={pp:.4f}  (mean over conditions; one shared within-concept shuffle)")
        return obs, p

    print("=== 1. omnibus: does the call depend on the template (five templates)?")
    report("chi-square-type statistic", omnibus, list(range(5)), fmt="{:.2f}", n=5000)

    print("\n=== 2. congruence: template t vs unbiased, share of calls for t")
    report("average over noun/verb/adj templates", congruence, list(range(5)), n=5000)
    for k, t in enumerate(CATS):
        report(f"{t} template: share({t}) - share under unbiased", lambda S, k=k: cong_t(S, k), [0, TPL.index(t)])

    print("\n=== 3. specificity among the framed templates (null mean 1/3)")
    report("diagonal of template x call (noun/verb/adj templates)", specificity, FR, fmt="{:.3f}")
    for k, t in enumerate(CATS):
        report(f"{t} template: share({t}) minus the other two framed templates' share({t})",
               lambda S, k=k: spec_t(S, k), FR)
    print("\n  per probe (pooled over conditions):")
    for q, probe in enumerate(("T1", "T2", "T3")):
        Sq = [np.stack([(X[:, :, q:q + 1] == k).mean(-1) for k in range(3)], -1) for X in Xs]
        obs = np.array([specificity(S) for S in Sq]); tot = obs.sum(); ge = 0
        for _ in range(5000):
            perms = [rng.permutation(3) for _ in range(30)]
            ge += sum(specificity(perm_within(S, FR, perms)) for S in Sq) >= tot - 1e-12
        print(f"    {probe}: " + "  ".join(f"{nm} {o:.3f}" for nm, o in zip(names, obs))
              + f"  | pooled {obs.mean():.3f} p={(ge + 1) / 5001:.4f}")

    print("\n=== 4. mixed vs unbiased: total variation distance of the call distributions")
    report("TV(mixed, unbiased)", mixed_tv, [0, 4], fmt="{:.3f}", n=5000)

    print("\n=== heterogeneity of specificity across conditions (concept-level scores)")
    sc = np.array([[np.mean([S[i, FR[k], k] for k in range(3)]) for S in Ss] for i in range(30)])
    st, p = friedmanchisquare(*sc.T)
    print("  means: " + "  ".join(f"{nm} {v:.3f}" for nm, v in zip(names, sc.mean(0)))
          + f"\n  Friedman chi2={st:.2f}, p={p:.3f}")


if __name__ == "__main__":
    main()
