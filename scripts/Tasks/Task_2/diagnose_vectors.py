#!/usr/bin/env python
"""Why Task 2 reads no concept POS off the trained vectors: two offline checks.

Uses only saved surprisal matrices, on the frozen two-way probe (20 slots):
  surprisal_<m>.json      real words (the calibration words the fits came from)
  bin_surprisal_<m>.json  the 160 two-way trained vectors
  ctrl_surprisal_<m>.json injected real-word rows and random vectors

Check 1 -- are the trained vectors inside the range the decision rule was fitted
on? Per slot, each item's surprisal as a z-score against the real words' own
distribution in that slot, plus how much of each vector's decision is carried by
its single largest log-likelihood-ratio term.

Check 2 -- does a decision rule with no fitted Gaussians recover concept POS?
Per slot, the baseline is the random vectors' mean and SD there. A vector's
"gain" in a slot is how much less surprising it makes the licensed continuation
than a random vector does, in random-SD units; noun score = mean gain over noun
slots, verb score = mean gain over verb slots, and D = noun - verb. Random
vectors average D = 0 by construction, so 0 is the threshold. The rule is checked
on the injected real words first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent / "out"


def llr(x, f):
    def lp(x, mu, s):
        s = max(s, 1e-6)
        return -0.5 * math.log(2 * math.pi * s * s) - (x - mu) ** 2 / (2 * s * s)
    return lp(x, f["mu_match"], f["sigma_match"]) - lp(x, f["mu_nonmatch"], f["sigma_nonmatch"])


def main() -> None:
    for m in ("gemma", "qwen", "aya"):
        probe = json.loads((HERE / f"bin2_probe_set_{m}.json").read_text())["slots"]
        slots = probe["noun"] + probe["verb"]
        spos = ["noun"] * len(probe["noun"]) + ["verb"] * len(probe["verb"])
        fits = json.loads((HERE / f"bin2_calibration_{m}.json").read_text())["fits"]
        real = json.loads((HERE / f"surprisal_{m}.json").read_text())
        calw = [w for p in ("noun", "verb") for w in real["words"]["calibration"][p]]
        vec = json.loads((HERE / f"bin_surprisal_{m}.json").read_text())["surprisal"]
        ctl = json.loads((HERE / f"ctrl_surprisal_{m}.json").read_text())["surprisal"]

        def mat(src, labels):
            return np.array([[src[k].get(l, np.nan) for k in slots] for l in labels], float)

        R = mat(real["surprisal"], calw)
        tl = sorted({l for k in slots for l in vec[k]})
        T = mat(vec, tl)
        cl = sorted({l for k in slots for l in ctl[k]})
        il = [l for l in cl if l.startswith("real_") and l.split("_")[1] in "nv"]
        rl = [l for l in cl if l.startswith("rand")]
        I, Z = mat(ctl, il), mat(ctl, rl)

        print(f"\n################ {m} ################")
        # ---- check 1: range
        mu, sd = np.nanmean(R, 0), np.nanstd(R, 0)
        print("检查 1：相对真实词分布的位置（每槽 z，所有槽合并）")
        print(f"  {'组':10}{'n':>5}{'中位 z':>9}{'中位|z|':>9}{'|z|>3 比例':>11}{'最大单槽贡献占比':>16}")
        for name, X in (("真实词", R), ("注入真实词", I), ("随机向量", Z), ("训练向量", T)):
            z = (X - mu) / sd
            L = np.array([[llr(X[i, j], fits[slots[j]]) for j in range(len(slots))] for i in range(len(X))])
            share = np.nanmax(np.abs(L), 1) / np.nansum(np.abs(L), 1)
            print(f"  {name:10}{len(X):>5}{np.nanmedian(z):>9.2f}{np.nanmedian(np.abs(z)):>9.2f}"
                  f"{np.nanmean(np.abs(z) > 3):>11.3f}{np.nanmedian(share):>16.3f}")
        zt = (T - mu) / sd
        print("  训练向量逐槽中位 z: noun 槽 " + " ".join(f"{v:+.1f}" for v in np.nanmedian(zt[:, :10], 0))
              + " | verb 槽 " + " ".join(f"{v:+.1f}" for v in np.nanmedian(zt[:, 10:], 0)))

        # ---- check 2: random-baseline rule
        b, s = np.nanmean(Z, 0), np.nanstd(Z, 0)
        def D(X):
            g = (b - X) / s
            return np.nanmean(g[:, :10], 1) - np.nanmean(g[:, 10:], 1)
        print("检查 2：不用高斯拟合，按随机向量基线判定（D>0 判 noun）")
        dI = D(I); yI = np.array([l.split("_")[1] for l in il])
        acc = np.mean((dI > 0) == (yI == "n"))
        print(f"  注入真实词 准确率 {acc:.3f}  (noun {np.mean(dI[yI=='n']>0):.2f}, verb {np.mean(dI[yI=='v']<=0):.2f})")
        dZ = D(Z); print(f"  随机向量 判 noun {np.mean(dZ>0):.2f}")
        dT = D(T); yT = np.array([l.split("_")[1] for l in tl]); lang = np.array([l[:2] for l in tl])
        for lg in ("en", "zh", "all"):
            sel = np.ones(len(tl), bool) if lg == "all" else lang == lg
            n_, v_ = sel & (yT == "n"), sel & (yT == "v")
            t = [[int((dT[n_] > 0).sum()), int((dT[n_] <= 0).sum())],
                 [int((dT[v_] > 0).sum()), int((dT[v_] <= 0).sum())]]
            _, pf = fisher_exact(t)
            accT = (t[0][0] + t[1][1]) / sel.sum()
            print(f"  训练向量[{lg}] 准确率 {accT:.3f}  概念n判noun {t[0][0]}/{sum(t[0])}  "
                  f"概念v判noun {t[1][0]}/{sum(t[1])}  差 {100*(t[0][0]/sum(t[0])-t[1][0]/sum(t[1])):.1f}pt  p={pf:.3f}")
        # concept-level test on the continuous D
        conc = np.array([l.split("_")[1] + "_" + l.split("_")[2] for l in tl])
        cm = {c: dT[conc == c].mean() for c in set(conc)}
        cn = [v for c, v in cm.items() if c[0] == "n"]; cv = [v for c, v in cm.items() if c[0] == "v"]
        u = mannwhitneyu(cn, cv, alternative="two-sided")
        print(f"  概念层（{len(cn)}+{len(cv)} 个概念，D 取平均）: 名词概念 D {np.mean(cn):+.3f}  "
              f"动词概念 D {np.mean(cv):+.3f}  AUC {u.statistic/(len(cn)*len(cv)):.2f}  p={u.pvalue:.3f}")
        print(f"  训练向量整体 D 均值 {dT.mean():+.3f}（随机向量为 0）")


if __name__ == "__main__":
    main()
