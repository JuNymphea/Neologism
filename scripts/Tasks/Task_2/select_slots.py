#!/usr/bin/env python
"""Screen the corpus candidate pool and freeze the final probe set.

    corpus candidates  ->  model-diagnostic screening  ->  10 N + 10 V + 10 A  ->  freeze

Corpus induction says which environments are POS-selective in real English.
It cannot say whether *this* model's continuation probabilities can read that
selectivity. This script answers the second question, using known words only.

The rule, fixed in advance:

    Walk each category's candidates in corpus-ranked order. Accept a candidate
    if its AUC on the **probe-dev** words reaches the threshold; otherwise skip
    it and look at the next one. Stop at ten.

Three disciplines make this instrument development rather than cherry-picking,
and all three matter:

**The threshold is identical for all three categories.** If adjectives need
the 3rd, 7th, 11th and 18th candidate to reach ten, that is fine -- they all
met the same corpus criterion and the same diagnosticity criterion. Adjectives
simply have to be searched deeper. Lowering the bar for them instead would make
the categories incomparable.

**A candidate with AUC below 0.5 is rejected, never flipped.** An AUC of 0.265
inverts to 0.735, but the probe's premise is that a POS-matched word makes the
licensed continuation *easier* to predict. A frame that does the opposite
contradicts that premise; it is not a usable signal in reverse.

**If the pool runs out before ten, nothing is forced.** That result means the
balanced ten-slot design is not achievable for this model with these frames --
which calls for a smaller balanced set (5/5/5, 8/8/8) or a different Task 2
method, not a relaxed bar for whichever category fell short.

Once frozen, the set does not change: not on the final-test words, and not on
anything the neologisms do.

    python Task_2/select_slots.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List

POS_KEYS = ("noun", "verb", "adj")


def auc(matched: List[float], nonmatched: List[float]) -> float:
    """P(a matched word is less surprising than a non-matched one).

    Rank-sum rather than thresholding: no distributional assumption, and
    comparable across slots whose surprisal scales differ widely.
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


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surprisal", type=Path, default=here / "out" / "surprisal.json")
    ap.add_argument("--out", type=Path, default=here / "out" / "probe_set.json")
    ap.add_argument("--n-slots", type=int, default=10, help="per POS")
    ap.add_argument("--min-auc", type=float, default=0.60,
                    help="identical for all three categories; never relaxed")
    args = ap.parse_args()

    data = json.loads(args.surprisal.read_text())
    S = data["surprisal"]
    words = data["words"]
    order = data.get("candidate_order") or list(S)
    meta = {f"{s['pos']}::{s['signature']}": s for s in data["slots"]}

    probedev = {p: words["probedev"][p] for p in POS_KEYS}

    def slot_auc(key: str, pos: str, pool) -> float:
        m = [S[key][w] for w in pool[pos]
             if w in S[key] and not math.isnan(S[key][w])]
        n = [S[key][w] for p in POS_KEYS if p != pos for w in pool[p]
             if w in S[key] and not math.isnan(S[key][w])]
        return auc(m, n)

    chosen: Dict[str, List[str]] = {p: [] for p in POS_KEYS}
    log: Dict[str, List[dict]] = {p: [] for p in POS_KEYS}
    for pos in POS_KEYS:
        for rank, key in enumerate([k for k in order if k.startswith(f"{pos}::")], 1):
            if len(chosen[pos]) >= args.n_slots:
                break
            a = slot_auc(key, pos, probedev)
            ok = (not math.isnan(a)) and a >= args.min_auc
            # Never invert a sub-0.5 frame into a "useful" one: it contradicts
            # the probe's premise rather than carrying reversed information.
            log[pos].append({
                "corpus_rank": rank, "signature": key.split("::")[1],
                "auc_probedev": round(a, 4) if not math.isnan(a) else None,
                "accepted": ok, "example": meta[key]["example"],
            })
            if ok:
                chosen[pos].append(key)

    print(f"阈值 AUC >= {args.min_auc}（三类相同，绝不放宽）\n")
    for pos in POS_KEYS:
        seen = len(log[pos])
        print(f"{pos.upper()}  取到 {len(chosen[pos])}/{args.n_slots}，"
              f"检视了语料排名前 {seen} 个候选")
        for e in log[pos]:
            mark = "✓" if e["accepted"] else " "
            a = f"{e['auc_probedev']:.3f}" if e["auc_probedev"] is not None else "  -  "
            print(f"  {mark} #{e['corpus_rank']:<3} AUC {a}  {e['example'][:44]}")
        short = args.n_slots - len(chosen[pos])
        if short > 0:
            print(f"  !! 候选池耗尽，仍缺 {short} 个 —— 不强凑。"
                  f"平衡的 {args.n_slots} 槽设计对该词性不可达。")
        print()

    complete = all(len(chosen[p]) == args.n_slots for p in POS_KEYS)
    aucs = {p: [e["auc_probedev"] for e in log[p] if e["accepted"]] for p in POS_KEYS}
    print(f"{'':6}{'slots':>7}{'平均 AUC':>11}{'最低':>8}")
    for pos in POS_KEYS:
        v = aucs[pos]
        print(f"{pos:<6}{len(v):>7}{st.mean(v) if v else float('nan'):>11.3f}"
              f"{min(v) if v else float('nan'):>8.3f}")
    print(f"\n{'冻结完成' if complete else '未达成平衡设计 —— 见上方提示'}")
    if complete:
        print("此后不得再更换 slot：不依据 final-test，更不依据 neologism 结果。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": data.get("model"),
        "n_slots_per_pos": args.n_slots,
        "min_auc": args.min_auc,
        "screened_on": "probe-dev control words",
        "balanced_design_achieved": complete,
        "slots": chosen,
        "mean_auc_probedev": {p: (round(st.mean(aucs[p]), 4) if aucs[p] else None)
                              for p in POS_KEYS},
        "screening_log": log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
