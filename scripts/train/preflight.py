"""
Preflight check against the REAL model, on a GPU node.

test_memory_opt.py verifies the maths on a tiny randomly-initialised Gemma3. This
covers what that cannot: the actual gemma-3-4b checkpoint, the installed transformers
version, and measured (rather than estimated) GPU memory.

  1. model class and the hidden-state path used to avoid materializing logits
  2. Gemma's embedding scale, checked against the model instead of assumed
  3. the wrapper's output vs writing the vector into the embedding matrix
  4. a few real training steps on the LONGEST batch in the dataset, with the peak
     memory that actually got allocated
  5. save -> reload round trip

Run:  python preflight.py --concept a_25
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from train_neologism import (
    DEFAULT_NEW_TOKEN,
    NeologismDataset,
    NewTokenEmbedding,
    _transformer_body,
    apo_up_loss,
    collate_fn,
    load_new_token_embedding,
    new_token_init_vector,
    save_new_token_embedding,
)

GB = 1024 ** 3
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="neologism/model/google/gemma-3-4b-it")
    ap.add_argument("--concept", default="a_25")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--template", default="verb")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        print("no GPU visible -- run this inside a GPU job")
        return 1

    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / GB
    print(f"\n=== environment ===")
    print(f"  torch {torch.__version__} | transformers "
          f"{__import__('transformers').__version__}")
    print(f"  {torch.cuda.get_device_name(0)}  {total:.1f} GB\n")

    print("=== model ===")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16
    ).to(dev)
    weights_mem = torch.cuda.memory_allocated() / GB
    print(f"  class: {type(model).__name__}, weights on GPU: {weights_mem:.2f} GB")

    # 1. hidden-state path -----------------------------------------------------
    try:
        body = _transformer_body(model)
        out = body(input_ids=torch.tensor([[1, 2, 3]], device=dev),
                   attention_mask=torch.ones(1, 3, dtype=torch.long, device=dev),
                   use_cache=False)
        h = out.last_hidden_state
        check("transformer body returns hidden states", h.shape[-1] == model.config.text_config.hidden_size
              if hasattr(model.config, "text_config") else h.shape[-1] > 0,
              f"{type(body).__name__} -> {tuple(h.shape)}")
    except Exception as e:
        check("transformer body returns hidden states", False, f"{type(e).__name__}: {e}")
        print("  (the fallback path would be used: logits get materialized, more memory)")

    # 2. embedding scale -------------------------------------------------------
    base = model.get_input_embeddings()
    scale = getattr(base, "embed_scale", None)
    check("embedding module exposes embed_scale", scale is not None,
          f"{type(base).__name__}, scale={float(scale) if scale is not None else 'MISSING'}")

    # --- set up exactly as training does --------------------------------------
    if DEFAULT_NEW_TOKEN in tok.get_vocab():
        raise ValueError("token already in vocab")
    tok.add_tokens([DEFAULT_NEW_TOKEN])
    new_id = tok.convert_tokens_to_ids(DEFAULT_NEW_TOKEN)
    if new_id >= base.weight.size(0):
        model.resize_token_embeddings(len(tok))
        base = model.get_input_embeddings()

    gen = torch.Generator().manual_seed(args.seed)
    init_vec = new_token_init_vector(model, new_id, init_mode="random", generator=gen)

    for p in model.parameters():
        p.requires_grad = False
    new_emb = NewTokenEmbedding(base, new_id, init_vec)
    model.set_input_embeddings(new_emb)

    # 3. wrapper vs writing the row, on the real model -------------------------
    ids = torch.tensor([[new_id]], device=dev)
    with torch.no_grad():
        got = model.get_input_embeddings()(ids)[0, 0].float()
        saved_row = base.weight[new_id].clone()
        base.weight[new_id] = new_emb.new_vec.to(base.weight.dtype)
        want = base(ids)[0, 0].float()          # through Gemma's own scaled embedding
        base.weight[new_id] = saved_row
    absd = (got - want).abs().max().item()
    reld = absd / max(want.abs().max().item(), 1e-9)
    # The two paths round to bf16 at different points: the wrapper keeps the vector in
    # fp32 until after the scale, writing the row rounds before it. They therefore
    # differ by about one bf16 ulp (0.4%), which an absolute threshold cannot tell
    # apart from a real error. A missing scale would show up as ~98%.
    check("wrapper == writing the row (scale applied)", reld < 0.02,
          f"rel {reld:.3%}, abs {absd:.2e}  (one bf16 ulp is ~0.39%)")

    # 4. real training steps on the worst-case batch ---------------------------
    print("\n=== training steps (longest batch in the dataset) ===")
    ds = NeologismDataset(concept=args.concept, tokenizer=tok,
                          new_token=DEFAULT_NEW_TOKEN, template=args.template,
                          seed=args.seed, data_dir=args.data_dir)
    lengths = [len(ds[i]["prompt_input_ids"]) + max(len(ds[i]["chosen_input_ids"]),
                                                    len(ds[i]["rejected_input_ids"]))
               for i in range(len(ds))]
    worst = sorted(range(len(ds)), key=lambda i: -lengths[i])[:args.batch_size]
    batch = collate_fn([ds[i] for i in worst], pad_token_id=pad_id)
    T = batch["prompt_input_ids"].size(1) + max(batch["chosen_input_ids"].size(1),
                                                batch["rejected_input_ids"].size(1))
    print(f"  dataset {len(ds)} rows | worst-case T = {T} tokens | batch = {args.batch_size}")

    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW([new_emb.new_vec], lr=args.lr)

    torch.cuda.reset_peak_memory_stats()
    before = new_emb.new_vec.detach().clone()
    for step in range(args.steps):
        loss = apo_up_loss(model, new_emb, batch, pad_id, args.beta, args.chunk_size)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"  step {step + 1}: loss = {loss.item():.4f} | "
              f"peak = {torch.cuda.max_memory_allocated() / GB:.2f} GB")

    peak = torch.cuda.max_memory_allocated() / GB
    moved = (new_emb.new_vec.detach() - before).norm().item()
    check("loss is finite", torch.isfinite(loss).item(), f"{loss.item():.4f}")
    check("the vector actually moved", moved > 0, f"|delta| = {moved:.3e}")
    check("fits in this GPU", peak < total * 0.95,
          f"peak {peak:.2f} GB / {total:.1f} GB ({100 * peak / total:.0f}%)")

    # 5. save -> reload --------------------------------------------------------
    path = "/tmp/preflight_embedding.pt"
    save_new_token_embedding(new_emb, DEFAULT_NEW_TOKEN, path)
    payload = torch.load(path, map_location="cpu")
    check("saved payload round-trips",
          torch.allclose(payload["embedding"], new_emb.new_vec.detach().float().cpu()),
          f"{payload['embedding'].numel()} floats, {payload['new_token']}")

    print(f"\nheadroom: peak {peak:.2f} GB of {total:.1f} GB -> "
          f"batch_size could go to roughly {int(args.batch_size * (total * 0.9) / peak)}")
    ok = all(results)
    print("\n" + ("preflight passed -- safe to submit training" if ok else "PREFLIGHT FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
