#!/usr/bin/env python
"""Remove the random init from trained vectors, keep what training learned.

Each trained vector is v = init + Delta: init is the N(0, 0.02) start (shared by
every seed-42 cell of a model), Delta is what training added. Templates leave a
strong trace in Delta in all three models (template identity decodable at
0.97-0.99), yet only qwen's vectors are read as the template's part of speech.
Two readings: the trace is POS-like everywhere but the off-manifold init masks it
in gemma and aya; or gemma's and aya's traces are simply not POS-like. Swapping the
init out separates them:

  mu     mean real-word row + Delta   (Delta placed at the centre of the word cloud)
  muw    mean real-word row + Delta scaled to a typical word's deviation from the mean
  delta  Delta alone

both rescaled to the model's median real-word row norm (length was shown not to
matter, so this only keeps the scale familiar).
"""
import argparse
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--vectors", type=Path, required=True, help="vectors_<m>.npz: V, I, label, lang, concept, template")
ap.add_argument("--realwords", type=Path, required=True, help="realwords_<m>.npz: X")
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--lang", default="en")
ap.add_argument("--remote-prefix", default=None, help="path prefix to write into the spec files")
a = ap.parse_args()

z = np.load(a.vectors); X = np.load(a.realwords)["X"].astype(np.float32)
mu = X.mean(0); target = float(np.median(np.linalg.norm(X, axis=1)))
sel = z["lang"] == a.lang
V, I = z["V"][sel].astype(np.float32), z["I"][sel].astype(np.float32)
labels, concepts, templates = z["label"][sel], z["concept"][sel], z["template"][sel]
D = V - I
prefix = a.remote_prefix or str(a.out_dir)
# Delta scaled (one factor for all vectors of the model) to the size of a typical
# real word's deviation from the word mean, so that "mean + Delta" sits where a
# word would -- the ratio is 0.69 / 0.76 / 0.49 for gemma / qwen / aya otherwise.
dev = float(np.median(np.linalg.norm(X - mu, axis=1)))
k = dev / float(np.median(np.linalg.norm(D, axis=1)))
print(f"word deviation {dev:.3f}; Delta scale factor for muw {k:.2f}")
for variant, M in (("mu", mu[None, :] + D), ("muw", mu[None, :] + k * D), ("delta", D)):
    d = a.out_dir / variant; d.mkdir(parents=True, exist_ok=True)
    spec_all, spec_nv = [], []
    for lab, c, t, v in zip(labels, concepts, templates, M):
        v = v * (target / np.linalg.norm(v))
        torch.save({"new_token": "~jdsglmdh", "new_token_id": -1,
                    "embedding": torch.from_numpy(v.copy()), "variant": variant}, d / f"{lab}.pt")
        line = f"{lab}={prefix}/{variant}/{lab}.pt"
        spec_all.append(line)
        if c[0] in "nv" and t != "adj":
            spec_nv.append(line)
    (d / "all.txt").write_text("\n".join(spec_all) + "\n")
    (d / "nv.txt").write_text("\n".join(spec_nv) + "\n")
    cos_mu = float(np.median((M @ mu) / np.linalg.norm(M, axis=1) / np.linalg.norm(mu)))
    Xu = X / np.linalg.norm(X, axis=1, keepdims=True); Mu = M / np.linalg.norm(M, axis=1, keepdims=True)
    nn = float(np.median((Mu @ Xu.T).max(1)))
    print(f"{variant:6} {len(spec_all)} vectors -> norm {target:.3f}; median cos to word mean {cos_mu:.3f}; "
          f"median cos to nearest real word {nn:.3f}")
Vu = V / np.linalg.norm(V, axis=1, keepdims=True)
print(f"(original: cos to word mean {float(np.median(Vu @ mu / np.linalg.norm(mu))):.3f}; "
      f"cos to nearest real word {float(np.median((Vu @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T).max(1))):.3f})")
