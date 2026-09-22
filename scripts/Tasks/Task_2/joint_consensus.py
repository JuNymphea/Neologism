#!/usr/bin/env python
"""One fixed probe for all three models, chosen for being repeatedly chosen.

Design C asks for a single slot set the three models share. A joint search
finds one, but not *the* one: the objective has a plateau, and which point on
it a run reports is decided by the shuffle order. Four seeds on aya all reached
probe-dev 0.9820 while agreeing on only 17 to 23 of 30 slots. That is fine for
estimating accuracy and useless for publishing an instrument -- the ten slots
would be one run's accident, with no answer to "why these ten".

So search from many seeds and keep the slots that keep coming back. A slot in
nearly every optimum is doing work no other candidate does; a slot in one is an
interchangeable member of an equivalence class. The result is deterministic
given the seed range, and its cost is reported against the best single run.

The search is vectorised because consensus needs many runs: the accuracy of a
subset is a couple of matrix operations per model, not a Python loop over
words, which is what makes twenty seeds practical rather than an overnight job.

    python Task_2/joint_consensus.py --seeds 20 --restarts 40
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np

from select_slots import POS_KEYS, Probe

MODELS = ("gemma", "qwen", "aya")


class Fast:
    """A model's probe-dev evaluation as matrices.

    `R[i, j]` is slot j's log-likelihood ratio for word i, already fitted on the
    calibration words. Accuracy for a subset is then a masked column sum per
    part of speech and an argmax -- no per-word Python.
    """

    def __init__(self, probe: Probe, pool, slots):
        words = [(POS_KEYS.index(g), w) for g in POS_KEYS for w in pool[g]]
        self.y = np.array([g for g, _ in words])
        self.R = np.array([[probe.r[k].get(w, np.nan) for k in slots]
                           for _, w in words])
        self.ok = ~np.isnan(self.R)
        self.R = np.nan_to_num(self.R)

    def accuracy(self, mask, slot_pos):
        s = np.zeros((len(self.y), 3))
        c = np.zeros((len(self.y), 3))
        for t in range(3):
            cols = mask & (slot_pos == t)
            if cols.any():
                s[:, t] = (self.R[:, cols] * self.ok[:, cols]).sum(1)
                c[:, t] = self.ok[:, cols].sum(1)
        mean = np.where(c > 0, s / np.maximum(c, 1), -np.inf)
        return float((mean.argmax(1) == self.y).mean())


def hill_climb(fasts, slot_pos, idx, n_slots, rng, rounds=80):
    """Swap-based climb on the weakest model's accuracy, mean as tie-break."""
    def score(mask):
        a = [f.accuracy(mask, slot_pos) for f in fasts]
        return (min(a), sum(a) / len(a))

    mask = np.zeros(len(slot_pos), bool)
    for t in range(3):
        for j in rng.sample(list(idx[t]), n_slots):
            mask[j] = True
    best = score(mask)
    for _ in range(rounds):
        moves = [(i, j) for t in range(3)
                 for i in np.flatnonzero(mask & (slot_pos == t))
                 for j in idx[t] if not mask[j]]
        rng.shuffle(moves)
        improved = False
        for i, j in moves:
            trial = mask.copy()
            trial[i], trial[j] = False, True
            s = score(trial)
            if s > best:
                mask, best, improved = trial, s, True
                break
        if not improved:
            break
    return best, mask


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--surprisal-dir", type=Path, default=here / "out")
    ap.add_argument("--n-slots", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--out", type=Path, default=here / "out" / "probe_set_C.json")
    args = ap.parse_args()

    probes, pdev, ftest, data = {}, {}, {}, {}
    for m in args.models:
        d = json.loads((args.surprisal_dir / f"surprisal_{m}.json").read_text())
        data[m] = d
        probes[m] = Probe(d, d["words"]["calibration"])
        pdev[m] = {p: d["words"]["probedev"][p] for p in POS_KEYS}
        ftest[m] = {p: d["words"]["finaltest"][p] for p in POS_KEYS}

    # Only slots every model could fit are eligible for a set they all share.
    cand = {p: sorted(set.intersection(*(set(probes[m].candidates[p])
                                         for m in args.models)))
            for p in POS_KEYS}
    slots = [k for p in POS_KEYS for k in cand[p]]
    slot_pos = np.array([POS_KEYS.index(k.split("::")[0].lower()) for k in slots])
    idx = {t: np.flatnonzero(slot_pos == t) for t in range(3)}
    print(f"共享候选：" + "  ".join(f"{p}={len(cand[p])}" for p in POS_KEYS))

    dev = [Fast(probes[m], pdev[m], slots) for m in args.models]
    fin = [Fast(probes[m], ftest[m], slots) for m in args.models]

    votes = collections.Counter()
    best_single = ((-1.0, -1.0), None)
    for seed in range(args.seeds):
        rng = random.Random(seed)
        run = max((hill_climb(dev, slot_pos, idx, args.n_slots, rng)
                   for _ in range(args.restarts)), key=lambda x: x[0])
        (worst, mean), mask = run
        votes.update(np.flatnonzero(mask).tolist())
        if run[0] > best_single[0]:
            best_single = run
        print(f"  seed {seed:>2}  最差 {worst:.4f}  平均 {mean:.4f}", flush=True)

    # Ties on votes fall to the higher worst-model accuracy of the slot alone,
    # so the rule is deterministic rather than dict-order dependent.
    def solo(j):
        m = np.zeros(len(slots), bool); m[j] = True
        return min(f.accuracy(m, slot_pos) for f in dev)

    chosen_idx = []
    for t in range(3):
        ranked = sorted(idx[t], key=lambda j: (-votes[j], -solo(j)))
        chosen_idx += ranked[:args.n_slots]
    mask = np.zeros(len(slots), bool)
    mask[chosen_idx] = True

    def report(label, m):
        d = [f.accuracy(m, slot_pos) for f in dev]
        f_ = [f.accuracy(m, slot_pos) for f in fin]
        print(f"  {label:<12}probe-dev 最差 {min(d):.4f}  "
              + "final-test " + "  ".join(f"{n} {v:.3f}" for n, v in zip(args.models, f_)))
        return f_

    print(f"\n=== 共识集 vs 单次最优 ===")
    f_cons = report("共识集", mask)
    f_best = report("单次最优", best_single[1])

    meta = probes[args.models[0]].meta
    print(f"\n=== 方法 C 的 {args.n_slots * 3} 个槽（票数 / {args.seeds}）===")
    out = {p: [] for p in POS_KEYS}
    for t, p in enumerate(POS_KEYS):
        print(f"\n  {p.upper()}")
        for j in sorted([j for j in chosen_idx if slot_pos[j] == t],
                        key=lambda j: -votes[j]):
            out[p].append(slots[j])
            print(f"    {votes[j]:>2}/{args.seeds}  {meta[slots[j]]['example'][:44]:<46}"
                  f"{meta[slots[j]]['diagnostic']}")
    tally = collections.Counter(votes[j] for t in range(3) for j in idx[t])
    print(f"\n  票数分布（全部 {len(slots)} 个候选）: "
          + "  ".join(f"{k}票×{v}" for k, v in sorted(tally.items(), reverse=True)[:8]))

    args.out.write_text(json.dumps({
        "selection": f"design C: joint search over {args.models}, "
                     f"consensus over {args.seeds} seeds",
        "models": args.models, "n_slots_per_pos": args.n_slots,
        "seeds": args.seeds, "restarts": args.restarts,
        "finaltest_consensus": dict(zip(args.models, [round(x, 4) for x in f_cons])),
        "finaltest_best_single": dict(zip(args.models, [round(x, 4) for x in f_best])),
        "votes": {slots[j]: votes[j] for j in chosen_idx},
        "slots": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
