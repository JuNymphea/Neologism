#!/usr/bin/env python
"""Screen the corpus candidate pool and freeze the final probe set.

    corpus candidates  ->  search on probe-dev words  ->  n slots per POS  ->  freeze

Corpus induction says which environments are POS-selective in real English. It
cannot say whether *this* model's continuation probabilities can read that
selectivity. This script answers the second question, using known words only.

**What is optimised is the probe's own accuracy, not each slot's AUC.** The
earlier rule walked the corpus ranking and kept any slot whose individual
probe-dev AUC cleared 0.60. That has two failure modes, and the frozen set it
produced had both:

  It stopped as soon as ten were found -- at corpus rank 11 for verbs, leaving
  29 of the 40 verb candidates never evaluated.

  Per-slot AUC selects slots that are individually strong, which in practice
  means slots that repeat each other. Nine of the ten verb slots asked for a
  direct object (`it` seven times over), so the set tested transitivity rather
  than verbhood and every intransitive verb -- succeed, sympathize, proceed --
  was read as a noun. Verb accuracy was 0.667 with the *highest* mean per-slot
  AUC of the three categories.

Searching over subsets fixes both. A slot that is weak alone can be the one
that catches what the others miss: the set found here includes `They can WORD
for` at AUC 0.301 and `They'll WORD again` at 0.320, which the 0.60 threshold
would never have considered, and which are exactly the frames intransitive
verbs sit in.

Three disciplines are unchanged, and all three matter:

**The search only ever sees the probe-dev words.** Fitting is on calibration,
selection on probe-dev, and final-test is left for `calibrate_probe.py` to read
once. Selecting on the words the accuracy is reported on would make that
accuracy meaningless for neologisms, which is the whole point of the probe.

**A candidate whose realized context cannot hold a bare-form control word is
excluded before scoring.** "All the WORD happened" wants a plural; every
control word is singular, so its surprisal there would measure the number
mismatch, not the category. The requirement is read off the realized prefix
rather than the frame's corpus features, because the two disagree: the corpus
says `NUM {SLOT} of` is usually plural, but realization fills the NUM with
"one", so the item actually wants a singular after all.

**The search is reported with its optimism.** Picking 30 slots out of 107 to
maximise accuracy on 222 words overfits them; the same protocol run inside
probe-dev estimates by how much.

Once frozen, the set does not change: not on the final-test words, and not on
anything the neologisms do.

**Which categories are being decided between is a parameter.** `--pos-keys
noun,verb` drops the adjective slots and searches for the best two-way set.
This is not cosmetic: each slot's non-match distribution is fitted on *the
other* categories (see `Probe.__init__`), so removing one re-fits every slot,
and a slot that earns its place by separating nouns from adjectives is dead
weight once adjectives are gone. `calibrate_probe.py` takes the same flag and
must be given the same value.

    python Task_2/select_slots.py
    python Task_2/select_slots.py --pos-keys noun,verb --n-slots 10
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Dict, List

POS_KEYS = ("noun", "verb", "adj")
UPPER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}

#: Determiners and quantifiers that fix the number of the noun they introduce.
#: Read off the *realized* prefix, not the frame's corpus features: the two can
#: disagree. `NUM {SLOT} of` is mostly plural in the corpus ("two dogs of"), but
#: realization fills the NUM with "one", so the item actually wants a singular.
SINGULAR_CUES = {"a", "an", "one", "this", "that", "each", "every", "another"}
PLURAL_CUES = {"all", "both", "these", "those", "many", "few", "several",
               "two", "three", "four", "five", "some", "most", "various"}


def incompatible_with_bare_form(prefix_tokens: List[str]) -> str | None:
    """Why a bare-form control word cannot go in this item, or None."""
    lowered = [t.lower() for t in prefix_tokens]
    try:
        slot = lowered.index("{neologism}")
    except ValueError:
        return None
    # Only the determiner region immediately before the slot binds its number.
    for tok in reversed(lowered[:slot]):
        if tok in PLURAL_CUES:
            return f"realized prefix requires a plural ('{tok}')"
        if tok in SINGULAR_CUES:
            return None
        if tok in {"the", "other", "different", "same", "very", "more", "most"}:
            continue
        break
    return None


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return -0.5 * math.log(2 * math.pi * sigma * sigma) - (x - mu) ** 2 / (2 * sigma * sigma)


def auc(matched: List[float], nonmatched: List[float]) -> float:
    """P(a matched word is less surprising than a non-matched one).

    Reported per slot for the record; it no longer decides anything.
    """
    if not matched or not nonmatched:
        return float("nan")
    merged = sorted([(s, 1) for s in matched] + [(s, 0) for s in nonmatched])
    ranks, i = {}, 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n1, n0 = len(matched), len(nonmatched)
    return 1.0 - (r - n1 * (n1 + 1) / 2) / (n1 * n0)


class Probe:
    """The per-slot log-likelihood ratios, and accuracy for any slot subset."""

    def __init__(self, data, pool_for_fit):
        self.S = data["surprisal"]
        self.meta = {f"{s['pos']}::{s['signature']}": s for s in data["slots"]}
        self.prefix = {f"{s['pos']}::{s['signature']}": s["prefix"].split()
                       for s in data["slots"]}
        self.order = data.get("candidate_order") or list(self.S)

        self.candidates: Dict[str, List[str]] = {}
        self.skipped: Dict[str, str] = {}
        for pos in POS_KEYS:
            keep = []
            for key in [k for k in self.order if k.startswith(UPPER[pos] + "::")]:
                why = incompatible_with_bare_form(self.prefix[key])
                if why:
                    self.skipped[key] = why
                    continue
                keep.append(key)
            self.candidates[pos] = keep

        # Fit each candidate on the calibration words, then precompute r for
        # every word once: the search evaluates thousands of subsets and must
        # not refit inside the loop.
        self.fits, self.r = {}, {}
        for pos in POS_KEYS:
            ok = []
            for key in self.candidates[pos]:
                m = [self.S[key][w] for w in pool_for_fit[pos]
                     if w in self.S[key] and not math.isnan(self.S[key][w])]
                n = [self.S[key][w] for p in POS_KEYS if p != pos
                     for w in pool_for_fit[p]
                     if w in self.S[key] and not math.isnan(self.S[key][w])]
                if len(m) < 2 or len(n) < 2:
                    continue
                f = {"pos": pos, "mu_match": st.mean(m), "sigma_match": st.stdev(m),
                     "mu_nonmatch": st.mean(n), "sigma_nonmatch": st.stdev(n),
                     "n_match": len(m), "n_nonmatch": len(n)}
                self.fits[key] = f
                self.r[key] = {
                    w: gaussian_logpdf(v, f["mu_match"], f["sigma_match"])
                       - gaussian_logpdf(v, f["mu_nonmatch"], f["sigma_nonmatch"])
                    for w, v in self.S[key].items() if not math.isnan(v)}
                ok.append(key)
            self.candidates[pos] = ok

    def accuracy(self, chosen: Dict[str, List[str]], pool) -> float:
        hit = tot = 0
        for gold in POS_KEYS:
            for w in pool[gold]:
                means = {}
                for p in POS_KEYS:
                    v = [self.r[k][w] for k in chosen[p] if w in self.r[k]]
                    if v:
                        means[p] = sum(v) / len(v)
                if means and max(means, key=means.get) == gold:
                    hit += 1
                tot += 1
        return hit / tot if tot else 0.0

    def diagnostics_count(self, chosen) -> int:
        """Distinct first tokens across the set -- the tie-break.

        Independent tests average better than repeats of one test, and the set
        the old rule produced had three distinct first tokens for verbs.
        """
        return sum(len({self.meta[k]["diagnostic"][0] for k in chosen[p]})
                   for p in POS_KEYS)

    def slot_auc(self, key, pos, pool) -> float:
        m = [self.S[key][w] for w in pool[pos]
             if w in self.S[key] and not math.isnan(self.S[key][w])]
        n = [self.S[key][w] for p in POS_KEYS if p != pos for w in pool[p]
             if w in self.S[key] and not math.isnan(self.S[key][w])]
        return auc(m, n)


def search(probe: Probe, pool, n_slots: int, restarts: int, seed: int,
           rounds: int = 60, quiet: bool = False):
    """Swap-based hill climbing from many random starts, exactly n per POS."""
    rng = random.Random(seed)
    best = (-1.0, None)
    for _ in range(restarts):
        cur = {p: rng.sample(probe.candidates[p], n_slots) for p in POS_KEYS}
        score = probe.accuracy(cur, pool)
        for _ in range(rounds):
            moves = [(p, i, k) for p in POS_KEYS
                     for i in range(n_slots)
                     for k in probe.candidates[p] if k not in cur[p]]
            rng.shuffle(moves)
            improved = False
            for p, i, k in moves:
                trial = {q: list(v) for q, v in cur.items()}
                trial[p][i] = k
                a = probe.accuracy(trial, pool)
                if a > score or (a == score and
                                 probe.diagnostics_count(trial) > probe.diagnostics_count(cur)):
                    cur, score, improved = trial, a, True
                    break
            if not improved:
                break
        if (score, probe.diagnostics_count(cur)) > (best[0], probe.diagnostics_count(best[1])
                                                    if best[1] else -1):
            best = (score, {p: list(v) for p, v in cur.items()})
    return best


def consensus(probe: Probe, pool, n_slots: int, restarts: int,
              n_seeds: int, rounds: int = 60):
    """The slots that keep being chosen, rather than one run's choice.

    Many subsets score the same on probe-dev -- four seeds on aya all reached
    0.9820 while sharing only 17 to 23 of 30 slots -- so the optimum a single
    run reports is one point on a plateau, and which point is decided by the
    shuffle order. That is fine for estimating accuracy and useless as an
    instrument: the probe has to be one fixed set, chosen for a reason.

    So run the search from several independent seeds and keep the slots that
    appear in most of the resulting optima. A slot selected every time is doing
    work no other candidate does; one selected once is an interchangeable
    member of an equivalence class. The result is deterministic given the seed
    range, and its probe-dev accuracy is reported alongside the best single
    run so the cost of insisting on stability is visible.
    """
    votes = {p: collections.Counter() for p in POS_KEYS}
    best_single = (-1.0, None)
    for seed in range(n_seeds):
        score, sel = search(probe, pool, n_slots, restarts, seed, rounds)
        for p in POS_KEYS:
            votes[p].update(sel[p])
        if score > best_single[0]:
            best_single = (score, sel)
        print(f"    seed {seed}: probe-dev {score:.4f}", flush=True)
    # Ties on vote count fall to the slot with the higher individual AUC, so
    # the rule stays deterministic rather than depending on Counter order.
    chosen = {}
    for p in POS_KEYS:
        ranked = sorted(votes[p].items(),
                        key=lambda kv: (-kv[1], -probe.slot_auc(kv[0], p, pool)))
        chosen[p] = [k for k, _ in ranked[:n_slots]]
    return chosen, votes, best_single


def optimism(data, n_slots, restarts, seed, quiet=False) -> float:
    """How much the search flatters itself, measured by the same protocol.

    probe-dev is split in half: search on one half, score on the other. The gap
    is what the final-test number should be expected to fall short by.
    """
    words = data["words"]
    rng = random.Random(seed + 1)
    gaps = []
    for rep in range(3):
        A = {p: [] for p in POS_KEYS}
        B = {p: [] for p in POS_KEYS}
        for p in POS_KEYS:
            ws = list(words["probedev"][p])
            rng.shuffle(ws)
            A[p], B[p] = ws[:len(ws) // 2], ws[len(ws) // 2:]
        for tr, te in ((A, B), (B, A)):
            probe = Probe(data, words["calibration"])
            s, sel = search(probe, tr, n_slots, max(8, restarts // 4), seed + rep, quiet=True)
            gaps.append(s - probe.accuracy(sel, te))
    return st.mean(gaps)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surprisal", type=Path, default=here / "out" / "surprisal.json")
    ap.add_argument("--out", type=Path, default=here / "out" / "probe_set.json")
    ap.add_argument("--n-slots", type=int, default=10, help="per POS")
    ap.add_argument("--pos-keys", default="noun,verb,adj",
                    help="要区分的类别；'noun,verb' 去掉形容词槽，改成二分类。"
                         "必须和 calibrate_probe.py 的 --pos-keys 一致")
    ap.add_argument("--restarts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--consensus", type=int, default=None, metavar="N",
                    help="稳定性选择：用 N 个种子各搜一次，取被选中次数最多的 10 个。"
                         "用于产出一把固定的尺子，而不是某一次搜索的结果")
    ap.add_argument("--skip-optimism", action="store_true",
                    help="跳过交叉验证的乐观偏差估计（省时间）")
    args = ap.parse_args()

    global POS_KEYS
    POS_KEYS = tuple(k.strip() for k in args.pos_keys.split(",") if k.strip())
    unknown = [p for p in POS_KEYS if p not in UPPER]
    if len(POS_KEYS) < 2 or unknown:
        ap.error(f"--pos-keys 需要至少两个来自 {sorted(UPPER)} 的类别"
                 + (f"，不认识 {unknown}" if unknown else ""))

    data = json.loads(args.surprisal.read_text())
    words = data["words"]
    probe = Probe(data, words["calibration"])
    probedev = {p: words["probedev"][p] for p in POS_KEYS}

    print(f"候选池（已排除形态不兼容的 {len(probe.skipped)} 个）："
          + "  ".join(f"{p}={len(probe.candidates[p])}" for p in POS_KEYS))
    print(f"在 probe-dev 的 {sum(len(v) for v in probedev.values())} 个词上搜索，"
          f"{args.restarts} 次随机重启\n")

    votes = None
    if args.consensus:
        print(f"稳定性选择：{args.consensus} 个种子各搜一次，取被选中次数最多的")
        chosen, votes, (best_score, _) = consensus(
            probe, probedev, args.n_slots, args.restarts, args.consensus)
        score = probe.accuracy(chosen, probedev)
        print(f"\n  共识集 probe-dev {score:.4f}   单次最好 {best_score:.4f}"
              f"   差 {score - best_score:+.4f}")
    else:
        score, chosen = search(probe, probedev, args.n_slots, args.restarts, args.seed)
    print(f"probe-dev {len(POS_KEYS)}分类准确率 {score:.4f}"
          f"   不同诊断首 token {probe.diagnostics_count(chosen)}")

    bias = None
    if not args.skip_optimism:
        bias = optimism(data, args.n_slots, args.restarts, args.seed)
        print(f"乐观偏差（probe-dev 内部对半，6 折）{bias:+.3f}"
              f"  → 留出预期约 {score - bias:.3f}")

    print(f"\n选中的 slot（AUC 为该槽单独在 probe-dev 上的判别力，仅供记录）")
    log = {}
    for pos in POS_KEYS:
        print(f"\n  {pos.upper()}")
        log[pos] = []
        for key in chosen[pos]:
            a = probe.slot_auc(key, pos, probedev)
            m = probe.meta[key]
            log[pos].append({"signature": key.split("::")[1], "auc_probedev": round(a, 4),
                             "diagnostic": m["diagnostic"], "example": m["example"]})
            flag = "✗" if a < 0.60 else " "
            print(f"   {a:>6.3f}{flag} {m['example'][:46]:<48}{m['diagnostic']}")
    low = sum(1 for p in POS_KEYS for e in log[p] if e["auc_probedev"] < 0.60)
    print(f"\n  其中 {low}/{args.n_slots * len(POS_KEYS)} 个单看 AUC 达不到 0.60 —— 旧的阈值规则不会考虑它们")
    for pos in POS_KEYS:
        c = collections.Counter(probe.meta[k]["diagnostic"][0] for k in chosen[pos])
        print(f"  {pos:<5} 诊断首 token {dict(c)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": data.get("model"),
        "n_slots_per_pos": args.n_slots,
        "pos_keys": list(POS_KEYS),
        "selection": f"hill-climbing on probe-dev {len(POS_KEYS)}-way accuracy",
        "screened_on": "probe-dev control words",
        "restarts": args.restarts, "seed": args.seed,
        "consensus_seeds": args.consensus,
        "vote_counts": ({p: dict(votes[p].most_common()) for p in POS_KEYS}
                        if votes else None),
        "probedev_accuracy": round(score, 4),
        "optimism_estimate": (round(bias, 4) if bias is not None else None),
        "balanced_design_achieved": all(len(chosen[p]) == args.n_slots for p in POS_KEYS),
        "distinct_diagnostics": probe.diagnostics_count(chosen),
        "slots": chosen,
        "selected_log": log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n冻结完成 -> {args.out}")
    print("此后不得再更换 slot：不依据 final-test，更不依据 neologism 结果。")


if __name__ == "__main__":
    main()
