#!/usr/bin/env python
"""Rescale trained vectors to a target norm, for the probes to read again.

The trained vectors sit at +10.1 SD (gemma), +1.6 SD (qwen), -1.8 SD (aya) from
each model's own real-word row norms, and qwen -- the only model whose vectors
sit inside its real-word range -- is the only one whose vectors show the
training-template effect. Direction is random-like in all three (cosine to the
nearest real word 0.06-0.10, random 0.06). So: pull gemma's and aya's vectors to
their real-word median norm, push qwen's out to +10 SD, and probe again. Only
the length changes; the direction, and so whatever the template wrote into it,
is untouched.
"""
import argparse, csv, json
from pathlib import Path
import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--manifest", type=Path, required=True)
ap.add_argument("--model-key", required=True)
ap.add_argument("--lang", default="en")
ap.add_argument("--target", type=float, required=True)
ap.add_argument("--out-dir", type=Path, required=True)
a = ap.parse_args()
a.out_dir.mkdir(parents=True, exist_ok=True)
spec_all, spec_nv, norms = [], [], []
for r in csv.DictReader(open(a.manifest, encoding="utf-8")):
    if r["model"] != a.model_key or r["lang"] != a.lang:
        continue
    d = torch.load(r["best_file"], map_location="cpu")
    v = d["embedding"].float(); norms.append(float(v.norm()))
    d["embedding"] = v * (a.target / v.norm())
    d["rescaled_from_norm"] = norms[-1]; d["rescaled_to"] = a.target
    lab = f"{r['lang']}_{r['concept']}_{r['template']}"
    p = a.out_dir / f"{lab}.pt"; torch.save(d, p)
    spec_all.append(f"{lab}={p}")
    if r["concept"][0] in "nv" and r["template"] != "adj":
        spec_nv.append(f"{lab}={p}")
(a.out_dir / "all.txt").write_text("\n".join(spec_all) + "\n")
(a.out_dir / "nv.txt").write_text("\n".join(spec_nv) + "\n")
print(f"{a.model_key} {a.lang}: {len(spec_all)} vectors, median norm {sorted(norms)[len(norms)//2]:.3f} -> {a.target}")
