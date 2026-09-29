#!/usr/bin/env python
"""Find Chinese Task 3 readings whose frame compatibility ties, and list them for
five more samples.

A vector's call is the frame (N/V/A) most of its synonyms fit; when two frames
fit equally often the call is undecided. Rather than split it, the tied vector
is sampled again -- five more generations with a fresh seed, pooled with the
earlier ones -- until one frame leads. Both the three-way tie (top of N/V/A) and
the two-way one (N against V, noun and verb concepts) count.

    python scripts/Tasks/Task_3/tiebreak_zh.py --round 1
writes Task_3/out/tiebreak/<file-stem>.round1.txt (LABEL=PATH per tied item)
for every Task 3 output with ties; submit_tiebreak_zh.slurm generates
<file-stem>_extra1.json next to each output, and analyze_zh_frames.py pools them.
"""
import argparse, glob, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "controls"))
import analyze_zh_frames as ZF  # noqa: E402

CTRL = "scripts/Tasks/controls"


def tied(c, idx):
    v = [c[i] for i in idx]; m = max(v)
    return sum(x >= m - 1e-9 for x in v) > 1


def paths_for(stem):
    """label -> vector path for one Task 3 output."""
    m = stem.rsplit("_", 1)[1]
    spec = (f"{CTRL}/legacy_zh_{m}/all.txt" if stem.startswith("legacyzh") else f"{CTRL}/zh_vec_specs/{m}_all.txt")
    out = dict(l.strip().split("=", 1) for l in open(HERE.parents[2] / spec, encoding="utf-8") if l.strip())
    return m, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    cache = json.load(open(HERE / "out/zh_frames_llm.json"))
    outdir = HERE / "out/tiebreak"; outdir.mkdir(exist_ok=True)
    total = 0
    for f in sorted(glob.glob(str(HERE / "out/*t3zh_synonyms_*.json"))):
        if "_extra" in f:
            continue
        stem = Path(f).stem
        C = ZF.compat(f, cache)
        keys = [k for k, c in C.items() if c is not None and
                (tied(c, [0, 1, 2]) or (k.split("_")[1] in "nv" and tied(c, [0, 1])))]
        unjudged = [w for v in ZF.load_samples(f).values() for s in v for w in s["synonyms"] if w.strip() not in cache]
        m, paths = paths_for(stem)
        lines = []
        for k in keys:
            if k.startswith("real_"):
                lines.append(f"{k}={CTRL}/vectors_zh_c3_{m}/{k}.pt")
            else:
                lines.append(f"{k}={paths[k]}")
        spec = outdir / f"{stem}.round{a.round}.txt"
        if lines:
            spec.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif spec.exists():
            spec.unlink()
        total += len(lines)
        print(f"{stem:34} tied {len(lines):3}" + (f"  (unjudged words: {len(unjudged)} -- run llm_frames_zh.py first)" if unjudged else ""))
    print(f"total tied: {total}")


if __name__ == "__main__":
    main()
