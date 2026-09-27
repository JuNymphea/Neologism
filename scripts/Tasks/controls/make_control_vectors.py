#!/usr/bin/env python
"""Vectors whose answer is known, written in the format the probes already read.

Both probes report a part of speech for a trained vector, and both are validated
on real words -- Task 1 at 0.982 to 0.990, Task 2 at 0.925 to 0.940. Neither
number licenses the vector results, because a real word in those prompts carries
its spelling: `-tion`, `-ly`, `-ate` are visible to the model whether or not it
can read anything off the embedding row. A trained vector has no spelling. The
probes are being asked a strictly harder question than the one they were
validated on, and nothing measured so far says they can answer it.

So ask them the same question with the answer known. Three sets, all written into
the same embedding row and scored through the identical path:

  real_<pos>_<word>   the embedding row of a real word, whose part of speech is
                      known, reached through the token `~jdsglmdh` so the surface
                      form carries nothing. A probe that reads part of speech off
                      an embedding should recover these; one that was living on
                      orthography cannot.

  randT_<i>           a random direction at the median norm of that model's
                      trained vectors. The empirical null: whatever the probes
                      report here is what they report for having no information
                      at that scale.

  randW_<i>           a random direction at the median norm of the real word rows
                      used above. Separates "the trained vectors are the wrong
                      length" from "a random direction at any length reads the
                      same" -- the trained vectors sit far off the real-word norm
                      (gemma +10.4 SD), and this says whether that matters.

The word's row is taken for its space-prefixed form, which is how a word occurs
in the probe prefixes ("The {NEOLOGISM} of"). A word that is not a single token
that way is skipped and named in the metadata rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--words", type=Path, required=True,
                    help="a control-word file: {'words': {pos: [...]}}")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--norm-trained", type=float, required=True,
                    help="median norm of this model's trained vectors")
    ap.add_argument("--n-random", type=int, default=50, help="per norm setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-keys", default="noun,verb,adj")
    ap.add_argument("--new-token", default="~jdsglmdh")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--form", default="space", choices=("space", "bare", "auto"),
                    help="which row: the space-prefixed form (how an English word sits in "
                         "the probe frames), the bare form (Chinese has no word spaces), "
                         "or auto (space if single-token, else bare)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pos_keys = [p.strip() for p in args.pos_keys.split(",") if p.strip()]
    short = {"noun": "n", "verb": "v", "adj": "a"}

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=None)
    emb = model.get_input_embeddings().weight.detach().to(torch.float32)
    print(f"embedding matrix {tuple(emb.shape)}", flush=True)

    words = json.loads(args.words.read_text())["words"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs, skipped, kept_norms = [], {}, []
    for pos in pos_keys:
        n_ok = 0
        for w in words[pos]:
            sp = tok.encode(" " + w, add_special_tokens=False)
            bare = tok.encode(w, add_special_tokens=False)
            if args.form == "space" or (args.form == "auto" and len(sp) == 1):
                ids = sp
            else:
                ids = bare
            if len(ids) != 1:
                skipped[w] = f"{len(sp)} tokens with space, {len(bare)} without"
                continue
            vec = emb[ids[0]].clone()
            path = args.out_dir / f"real_{short[pos]}_{w}.pt"
            torch.save({"new_token": args.new_token, "new_token_id": -1,
                        "embedding": vec, "source": f"row {ids[0]} = {w!r}"}, path)
            specs.append(f"real_{short[pos]}_{w}={path}")
            kept_norms.append(float(vec.norm()))
            n_ok += 1
        print(f"  {pos}: {n_ok}/{len(words[pos])} single-token", flush=True)

    kept_norms.sort()
    norm_word = kept_norms[len(kept_norms) // 2] if kept_norms else float("nan")
    print(f"real-word row norm: median {norm_word:.4f} "
          f"[{kept_norms[0]:.3f}, {kept_norms[-1]:.3f}]")
    print(f"trained-vector norm: median {args.norm_trained:.4f} "
          f"({args.norm_trained / norm_word:.2f}x the real-word median)")

    # Random directions, isotropic, then scaled. Isotropic is the right null for
    # "no information": any structured direction would be an assumption about
    # what an uninformative vector looks like.
    g = torch.Generator().manual_seed(args.seed)
    hidden = emb.shape[1]
    for tag, norm in (("randT", args.norm_trained), ("randW", norm_word)):
        for i in range(args.n_random):
            v = torch.empty(hidden).normal_(0.0, 1.0, generator=g)
            v = v / v.norm() * norm
            path = args.out_dir / f"{tag}_{i:03d}.pt"
            torch.save({"new_token": args.new_token, "new_token_id": -1,
                        "embedding": v, "source": f"random, norm {norm:.4f}"}, path)
            specs.append(f"{tag}_{i:03d}={path}")

    spec_file = args.out_dir / "spec.txt"
    spec_file.write_text("\n".join(specs) + "\n")
    (args.out_dir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "n_real": sum(1 for s in specs if s.startswith("real_")),
        "n_random_per_norm": args.n_random,
        "norm_trained_median": args.norm_trained,
        "norm_realword_median": norm_word,
        "norm_realword_range": [kept_norms[0], kept_norms[-1]] if kept_norms else None,
        "skipped_not_single_token": skipped,
        "seed": args.seed,
        "form": args.form,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(specs)} vectors -> {spec_file}")
    if skipped:
        print(f"skipped {len(skipped)} words that are not one token: "
              + ", ".join(list(skipped)[:8]) + ("..." if len(skipped) > 8 else ""))


if __name__ == "__main__":
    main()
