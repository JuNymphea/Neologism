#!/usr/bin/env python
"""Pull the input embedding rows for the Task 3 words, and nothing else.

Task 3 asks whether lexical category is linearly recoverable from the input
representation itself -- before any Transformer layer, without a prompt, a
continuation or a generation step. So the only thing needed from the model is
one row of the embedding table per word, which is why this reads the tensor
straight out of the safetensors shards rather than instantiating the model:
a 4B model's embedding matrix is about a gigabyte, the rest of it is dead
weight for this purpose, and the extraction then runs on CPU in seconds.

Every word must be a single token *after a space*, which is where the probes
put it. That already holds for the 975 shared words by construction; the check
stays because a wrong model path would otherwise produce a silently truncated
set rather than an error.

    python extract_embeddings.py --model <path> --name gemma
    python extract_embeddings.py --model <path> --name gemma \
        --extra-embeddings run1=/path/to/embedding_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POS_KEYS = ("noun", "verb", "adj")


def find_embedding_tensor(model_dir: Path):
    """The input embedding matrix, whatever the checkpoint calls it.

    Gemma 3 nests the language model under `language_model.`, so a hard-coded
    `model.embed_tokens.weight` misses it. Search instead, and refuse rather
    than guess when several candidates match.
    """
    from safetensors import safe_open

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{model_dir} 下没有 .safetensors")
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            keys = [k for k in f.keys()
                    if k.endswith("embed_tokens.weight") and "vision" not in k]
            if len(keys) > 1:
                raise SystemExit(f"找到多个候选，无法确定用哪个: {keys}")
            if keys:
                return shard, keys[0]
    raise SystemExit("没找到 embed_tokens.weight")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="模型目录（含 safetensors 与 tokenizer.json）")
    ap.add_argument("--name", required=True, help="模型简称，用于输出文件名")
    ap.add_argument("--tokenizer", type=Path, default=None, help="默认取 --model 下的 tokenizer.json")
    ap.add_argument("--splits", type=Path, default=here / "out" / "task3_splits.json")
    ap.add_argument("--out-dir", type=Path, default=here / "out" / "embeddings")
    ap.add_argument("--via-transformers", action="store_true",
                    help="改用 AutoModel 加载再取 embedding。慢、吃内存，但不依赖张量名，"
                         "用来核对直接读张量的结果")
    ap.add_argument("--extra-embeddings", nargs="*", default=[], metavar="LABEL=PATH",
                    help="额外的学习到的向量，如 neologism 的 .pt，形如 unbiased=/path/emb.pt")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer

    def torch_float32():
        return torch.float32

    tok_path = args.tokenizer or (args.model / "tokenizer.json")
    tok = Tokenizer.from_file(str(tok_path))
    splits = json.loads(args.splits.read_text())["splits"]
    words = sorted({w for s in splits.values() for p in POS_KEYS for w in s[p]})

    ids, missing = {}, []
    for w in words:
        enc = tok.encode(" " + w, add_special_tokens=False).ids
        (ids.setdefault(w, enc[0]) if len(enc) == 1 else missing.append(w))
    if missing:
        raise SystemExit(f"{len(missing)} 个词在 {args.name} 下不是单 token，例：{missing[:5]}")

    if args.via_transformers:
        # The path that needs no assumption about the tensor's name, at the
        # cost of materialising the whole model. Worth running once per model
        # to confirm the fast path agrees.
        from transformers import AutoModelForCausalLM
        print(f"{args.name}: 经 transformers 加载（较慢）")
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        W = mdl.get_input_embeddings().weight
        shape = tuple(W.shape)
        print(f"  embedding 矩阵 {shape[0]:,} × {shape[1]}")
        vecs = np.stack([W[ids[w]].detach().float().numpy() for w in words])
        key = "get_input_embeddings().weight"
    else:
        vecs, shape, key = None, None, None

    if vecs is None:
        shard, key = find_embedding_tensor(args.model)
        print(f"{args.name}: {key}  ({shard.name})")
    # Read through torch, not numpy: these tables are usually bfloat16, which
    # numpy has no dtype for. Slice row by row -- pulling the whole matrix in
    # would cost a gigabyte of RAM to keep 975 rows of it.
        with safe_open(shard, framework="pt") as f:
            E = f.get_slice(key)
            shape = E.get_shape()
            print(f"  embedding 矩阵 {shape[0]:,} × {shape[1]}")
            vecs = np.stack([E[ids[w]:ids[w] + 1, :].to(torch_float32()).numpy()[0]
                             for w in words])

    extra = {}
    for spec in args.extra_embeddings:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"用 label=path 的形式：{spec!r}")
        import torch
        v = torch.load(path, map_location="cpu")
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != shape[1]:
            raise SystemExit(f"{label}: 维度 {v.shape[0]} 与模型的 {shape[1]} 不符")
        extra[label] = v
        print(f"  额外向量 {label}: {v.shape[0]} 维")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.name}.npz"
    np.savez_compressed(out, words=np.array(words), vectors=vecs,
                        token_ids=np.array([ids[w] for w in words]),
                        dim=shape[1], model=str(args.model),
                        **{f"extra__{k}": v for k, v in extra.items()})
    n = np.linalg.norm(vecs, axis=1)
    print(f"  取出 {len(words)} 个词向量   范数 中位 {np.median(n):.3f}  "
          f"范围 {n.min():.3f}–{n.max():.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
