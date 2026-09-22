#!/usr/bin/env python
"""One slot set for several models at once, and what insisting on that costs.

Design B gave each model the slots its own probabilities read best. The three
sets overlapped on 12.7 of 30 slots, against 19.6 for two searches on the *same*
model with different seeds and 8.4 for a random draw -- so models really do
prefer different environments, but the preference is partial.

That leaves a question B cannot answer: is the preference load-bearing? If one
set is forced on all three and accuracy barely moves, the 59 slots the three
searches touched are largely interchangeable and the low overlap was a choice
among equivalents. If accuracy falls, each model depends on environments the
others cannot use.

Two measurements here:

**Cross-application.** Every model's own slot set, scored on every model. The
off-diagonal is what a borrowed instrument costs.

**Joint search.** One set chosen to maximise the *worst* model's probe-dev
accuracy, so no model is traded away for the others. Mean would let a strong
pair carry a weak one, which is the opposite of what a shared instrument is
for.

    python Task_2/joint_select.py
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics as st
from pathlib import Path

from select_slots import POS_KEYS, Probe, gaussian_logpdf

MODELS = ("gemma", "qwen", "aya")


def accuracy_on(probe: Probe, chosen, pool) -> float:
    return probe.accuracy(chosen, pool)


def joint_search(probes, pools, n_slots, restarts, seed, rounds=60):
    """Hill climb on the weakest model's accuracy, mean as the tie-break."""
    rng = random.Random(seed)
    # Only slots every model could fit are eligible: a candidate dropped for one
    # model cannot be in a set they all share.
    cand = {p: sorted(set.intersection(*(set(pr.candidates[p]) for pr in probes)))
            for p in POS_KEYS}

    def score(sel):
        accs = [accuracy_on(pr, sel, pool) for pr, pool in zip(probes, pools)]
        return (min(accs), st.mean(accs))

    best = ((-1.0, -1.0), None)
    for _ in range(restarts):
        cur = {p: rng.sample(cand[p], n_slots) for p in POS_KEYS}
        s = score(cur)
        for _ in range(rounds):
            moves = [(p, i, k) for p in POS_KEYS for i in range(n_slots)
                     for k in cand[p] if k not in cur[p]]
            rng.shuffle(moves)
            improved = False
            for p, i, k in moves:
                trial = {q: list(v) for q, v in cur.items()}
                trial[p][i] = k
                t = score(trial)
                if t > s:
                    cur, s, improved = trial, t, True
                    break
            if not improved:
                break
        if s > best[0]:
            best = (s, {p: list(v) for p, v in cur.items()})
    return best


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--surprisal-dir", type=Path, default=here / "out")
    ap.add_argument("--n-slots", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=here / "out" / "probe_set_joint.json")
    args = ap.parse_args()

    data, probes, pdev, final = {}, [], [], []
    for m in args.models:
        d = json.loads((args.surprisal_dir / f"surprisal_{m}.json").read_text())
        data[m] = d
        probes.append(Probe(d, d["words"]["calibration"]))
        pdev.append({p: d["words"]["probedev"][p] for p in POS_KEYS})
        final.append({p: d["words"]["finaltest"][p] for p in POS_KEYS})
    own = {m: json.loads((args.surprisal_dir / f"probe_set_{m}.json").read_text())["slots"]
           for m in args.models}

    # -- cross-application ---------------------------------------------------
    print("=== 用 A 的槽去测 B（final-test 准确率）===")
    print(f"{'槽来自':<10}" + "".join(f"{m:>9}" for m in args.models) + f"{'最差':>8}")
    for src in args.models:
        row = [accuracy_on(pr, own[src], f) for pr, f in zip(probes, final)]
        print(f"  {src:<8}" + "".join(f"{x:>9.3f}" for x in row) + f"{min(row):>8.3f}")
    print("  对角线是 B 方案，非对角线是借用别人的仪器")

    # -- joint search --------------------------------------------------------
    (worst, mean), sel = joint_search(probes, pdev, args.n_slots,
                                      args.restarts, args.seed)
    print(f"\n=== 联合搜索（目标：最差模型的 probe-dev 准确率）===")
    print(f"  probe-dev  最差 {worst:.4f}   平均 {mean:.4f}")
    print(f"\n{'':10}" + "".join(f"{m:>9}" for m in args.models))
    joint = [accuracy_on(pr, sel, f) for pr, f in zip(probes, final)]
    diag = [accuracy_on(pr, own[m], f) for pr, m, f in zip(probes, args.models, final)]
    print(f"  {'共用槽':<8}" + "".join(f"{x:>9.3f}" for x in joint))
    print(f"  {'各自槽':<8}" + "".join(f"{x:>9.3f}" for x in diag))
    print(f"  {'差':<9}" + "".join(f"{a-b:>+9.3f}" for a, b in zip(joint, diag)))

    meta = {f"{s['pos']}::{s['signature']}": s for s in data[args.models[0]]["slots"]}
    print(f"\n=== 共用的 {args.n_slots * 3} 个槽 ===")
    for p in POS_KEYS:
        print(f"  {p.upper()}")
        for k in sel[p]:
            used = [m for m in args.models if k in own[m][p]]
            mark = f"  ←B 里 {','.join(used)}" if used else ""
            print(f"    {meta[k]['example'][:44]:<46}{str(meta[k]['diagnostic']):<24}{mark}")
    overlap = {m: sum(len(set(sel[p]) & set(own[m][p])) for p in POS_KEYS)
               for m in args.models}
    print(f"\n  与各自 B 方案的重叠 /30: {overlap}")

    args.out.write_text(json.dumps({
        "selection": "joint hill-climbing on the worst model's probe-dev accuracy",
        "models": args.models, "n_slots_per_pos": args.n_slots,
        "probedev_worst": round(worst, 4), "probedev_mean": round(mean, 4),
        "finaltest_shared": dict(zip(args.models, [round(x, 4) for x in joint])),
        "finaltest_own": dict(zip(args.models, [round(x, 4) for x in diag])),
        "overlap_with_own": overlap,
        "slots": sel,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
