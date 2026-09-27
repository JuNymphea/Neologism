#!/usr/bin/env python
"""Does the template effect (and the concept-POS signal) grow with training?

Reads the trajectory run: every saved version of each multi-version cell through
Tasks 1-3. The seed-42 sequence ep1 -> ep3 -> ... -> ep17 is the trajectory
proper; the seed restarts (ep3_s43..47) and the lr-3e-3 runs (s3) are compared
against ep3_s42 / the same cell.
"""
import json, sys, collections, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Task_3"))
from analyze_synonyms import parse, Tagger, CATS  # noqa

HERE = Path(__file__).resolve().parent
meta = json.load(open(HERE / "traj/meta.json"))
tag = Tagger()
POS = {"n": "noun", "v": "verb", "a": "adj"}

def argmax(d): return {k: max(v, key=v.get) for k, v in d.items()}

def load(m):
    P = {}
    P["T1_3"] = argmax(json.load(open(HERE.parent / f"Task_1/out/traj3_{m}.json"))["probabilities"]["neologism"])
    P["T1_2"] = argmax(json.load(open(HERE.parent / f"Task_1/out/traj2_{m}.json"))["probabilities"]["neologism"])
    P["T2_3"] = json.load(open(HERE.parent / f"Task_2/out/traj3_calibration_{m}.json"))["neologisms"]
    P["T2_2"] = json.load(open(HERE.parent / f"Task_2/out/traj2_calibration_{m}.json"))["neologisms"]
    R = json.load(open(HERE.parent / f"Task_3/out/traj_synonyms_{m}.json"))["results"]
    P["T3_3"], P["T3_2"] = {}, {}
    for k, samples in R.items():
        acc = np.zeros(3); w = 0
        for s in samples:
            for x in parse(s["raw"]):
                d = tag(x)
                if d: acc += [d[c] for c in CATS]; w += 1
        p = acc / w if w else np.ones(3) / 3
        P["T3_3"][k] = CATS[int(np.argmax(p))]
        P["T3_2"][k] = "noun" if p[0] >= p[1] else "verb"
    return P

def dfrac(path):
    d = torch.load(path.replace("scripts/train/", str(HERE.parents[2] / "scripts/train/extra") + "/"), map_location="cpu")
    v = d["embedding"].float(); r = d.get("ref_embedding")
    if r is None:
        g = torch.Generator().manual_seed(int(d["seed"])); r = torch.empty(v.numel()).normal_(0, 0.02, generator=g)
    return float((v - r.float()).norm() / v.norm())

for m in ("gemma", "qwen", "aya"):
    P = load(m)
    items = {k.split("|", 1)[1]: v for k, v in meta.items() if v["model"] == m}
    print(f"\n{'#'*30} {m} {'#'*30}")
    for lang in ("en", "zh"):
        cells = collections.defaultdict(dict)
        for lab, v in items.items():
            if v["lang"] != lang: continue
            cells[(v["concept"], v["template"])][v["tag"]] = lab
        if not cells: continue
        print(f"\n===== {m} {lang}: {len(cells)} cells =====")
        # ---- seed-42 trajectory: pool all cells that have that epoch level
        levels = ["ep1_s42", "ep3_s42", "ep5_s42", "ep7_s42", "ep9_s42", "ep11_s42", "ep13_s42", "ep15_s42", "ep17_s42", "s3"]
        print(f"{'版本':9}{'n':>4}{'|Δ|/|v|':>8}  " + "".join(f"{t:>16}" for t in ("T1_3", "T2_3", "T3_3")) + "   |" + "".join(f"{t:>14}" for t in ("T1_2", "T2_2", "T3_2")))
        print(f"{'':9}{'':>4}{'':>8}  " + "".join(f"{'模版符/概念符/n·v·a':>16}" for _ in range(3)) + "   |" + "".join(f"{'模版符/概念符':>14}" for _ in range(3)))
        for lv in levels:
            labs = [cells[c][lv] for c in cells if lv in cells[c]]
            if not labs: continue
            df = np.median([dfrac(items[l]["path"]) for l in labs])
            row = f"{lv:9}{len(labs):>4}{df:>8.2f}  "
            for t in ("T1_3", "T2_3", "T3_3"):
                pr = [P[t][l] for l in labs]
                tm = [l for l in labs if items[l]["template"] in ("noun", "verb", "adj")]
                tmatch = np.mean([P[t][l] == items[l]["template"] for l in tm]) if tm else float("nan")
                cmatch = np.mean([P[t][l] == POS[items[l]["concept"][0]] for l in labs])
                c = collections.Counter(pr)
                row += f"{tmatch:.2f}/{cmatch:.2f}/{c['noun']}·{c['verb']}·{c['adj']}".rjust(16)
            row += "   |"
            labs2 = [l for l in labs if items[l]["concept"][0] in "nv" and items[l]["template"] != "adj"]
            for t in ("T1_2", "T2_2", "T3_2"):
                if not labs2: row += f"{'-':>14}"; continue
                tm = [l for l in labs2 if items[l]["template"] in ("noun", "verb")]
                tmatch = np.mean([P[t][l] == items[l]["template"] for l in tm]) if tm else float("nan")
                cmatch = np.mean([P[t][l] == POS[items[l]["concept"][0]] for l in labs2])
                row += f"{tmatch:.2f}/{cmatch:.2f}".rjust(14)
            print(row)
        # ---- paired: same cell, ep1_s42 vs its highest seed-42 epoch (>=5)
        pairs = []
        for c, vs in cells.items():
            hi = [t for t in vs if t.endswith("_s42") and t != "ep1_s42" and int(t[2:].split("_")[0]) >= 5]
            if "ep1_s42" in vs and hi:
                top = max(hi, key=lambda t: int(t[2:].split("_")[0]))
                pairs.append((c, vs["ep1_s42"], vs[top], top))
        if pairs:
            print(f"  配对（同一 cell，ep1 → 最高 epoch ≥5，n={len(pairs)}）：template 词性符合 ep1→高 / 概念词性符合 ep1→高")
            for t in ("T1_3", "T2_3", "T3_3"):
                tm = [(a, b) for (c, a, b, _) in pairs if c[1] in ("noun", "verb", "adj")]
                t1 = np.mean([P[t][a] == c1 for (a, b), c1 in zip(tm, [c[1] for (c, *_) in pairs if c[1] in ("noun", "verb", "adj")])])
                t2 = np.mean([P[t][b] == c1 for (a, b), c1 in zip(tm, [c[1] for (c, *_) in pairs if c[1] in ("noun", "verb", "adj")])])
                c1_ = np.mean([P[t][a] == POS[c[0][0]] for (c, a, b, _) in pairs]); c2_ = np.mean([P[t][b] == POS[c[0][0]] for (c, a, b, _) in pairs])
                print(f"    {t}: template {t1:.2f}→{t2:.2f}   concept {c1_:.2f}→{c2_:.2f}")
        # ---- seed restarts vs ep3_s42 on the same cells
        for alt in ("ep3_s43", "ep3_s44", "ep3_s45", "ep3_s46", "ep3_s47", "ep2_s43", "s3"):
            both = [(cells[c]["ep3_s42"], cells[c][alt]) for c in cells if alt in cells[c] and "ep3_s42" in cells[c]]
            if len(both) < 5: continue
            print(f"  {alt} vs ep3_s42（同一 cell，n={len(both)}）: 三任务预测一致率 "
                  + "  ".join(f"{t}:{np.mean([P[t][a]==P[t][b] for a,b in both]):.2f}" for t in ("T1_3", "T2_3", "T3_3")))
