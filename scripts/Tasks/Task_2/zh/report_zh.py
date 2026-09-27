#!/usr/bin/env python
"""Summarize the Chinese Task 2 probe sets: accuracy, optimism, and the frozen slots."""
import json
from pathlib import Path
O = Path(__file__).resolve().parents[1] / "out_zh"
EN = Path(__file__).resolve().parents[1] / "out"
cand = json.loads((O / "candidates.json").read_text())
item = {f"{p}::{c['signature']}": c for p in cand for c in cand[p]}
print(f"{'模型':7}{'probe-dev':>10}{'乐观偏差':>9}{'预期':>8}{'final-test':>11}{'noun':>7}{'verb':>7}   | 英文 final-test")
for m in ("gemma", "qwen", "aya"):
    try:
        ps = json.loads((O / f"probe_set_{m}.json").read_text())
        ca = json.loads((O / f"calibration_{m}.json").read_text())["accuracy"]["frozen_probe_set"]
    except FileNotFoundError:
        print(f"{m:7} (missing)"); continue
    en = json.loads((EN / f"bin2_calibration_{m}.json").read_text())["accuracy"]["frozen_probe_set"]["overall"]
    opt = ps.get("optimism_estimate") or 0
    print(f"{m:7}{ps['probedev_accuracy']:>10.4f}{opt:>9.4f}{ps['probedev_accuracy'] - opt:>8.4f}"
          f"{ca['overall']:>11.4f}{ca['per_pos']['noun']['accuracy']:>7.3f}{ca['per_pos']['verb']['accuracy']:>7.3f}"
          f"   | {en:.4f}")
for m in ("gemma", "qwen", "aya"):
    p = O / f"probe_set_{m}.json"
    if not p.exists():
        continue
    ps = json.loads(p.read_text())
    print(f"\n=== {m}: frozen slots ===")
    for pos in ("noun", "verb"):
        for k in ps["slots"][pos]:
            c = item.get(k, {})
            print(f"  {pos:5} {c.get('prefix', '?').replace('{NEOLOGISM}', '□')}{''.join(c.get('diagnostic', []))}"
                  f"    ({k.split('::')[1]}, purity {c.get('purity', float('nan')):.2f}, n {c.get('n', 0)})")
