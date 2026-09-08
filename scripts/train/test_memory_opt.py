"""
Check that the memory optimisations did not change the maths.

Builds a tiny randomly-initialised Gemma3 and compares, in fp32:

  1. chunked sequence_logps            vs  full log_softmax + gather
  2. NewTokenEmbedding                 vs  untie lm_head + write the row
  3. NewTokenEmbedding.reference_mode  vs  a separate model holding the init vector

Run:  python test_memory_opt.py
"""

import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Gemma3TextConfig, Gemma3ForCausalLM

from train_neologism import (
    NewTokenEmbedding,
    _final_hidden_states,
    _untie_output_embeddings,
    sequence_logps,
)

torch.manual_seed(0)

VOCAB, HIDDEN, NEW_ID = 128, 32, 120
B, T = 3, 17


def tiny_model():
    cfg = Gemma3TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, tie_word_embeddings=True,
    )
    return Gemma3ForCausalLM(cfg).to(torch.float32).eval()


def old_logps(model, input_ids, attention_mask, loss_mask):
    """The original implementation: materialise [B, T, V] and log_softmax over it."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    lp = lp.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * loss_mask[:, 1:].float()).sum(dim=-1)


def new_logps(model, input_ids, attention_mask, loss_mask, chunk_size=7):
    hidden = _final_hidden_states(model, input_ids, attention_mask)
    return sequence_logps(
        model, hidden[:, :-1, :], input_ids[:, 1:], loss_mask[:, 1:], chunk_size
    )


def report(name, a, b, tol=2e-4):
    diff = (a - b).abs().max().item()
    ok = diff < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} max|diff| = {diff:.3e}")
    return ok


def main():
    ids = torch.randint(0, VOCAB - 8, (B, T))
    ids[:, 3] = NEW_ID                                  # the neologism appears in the prompt
    attn = torch.ones_like(ids)
    mask = torch.zeros_like(ids)
    mask[:, 6:] = 1                                     # only the completion counts

    ok = True

    # --- 1. chunking only ------------------------------------------------------
    m = tiny_model()
    with torch.no_grad():
        ok &= report("chunked logps == full log_softmax",
                     new_logps(m, ids, attn, mask), old_logps(m, ids, attn, mask))
        for cs in (1, 5, 1000):
            ok &= report(f"chunk_size={cs} matches chunk_size=7",
                         new_logps(m, ids, attn, mask, cs), new_logps(m, ids, attn, mask))

    # --- 2. wrapper vs writing the row ----------------------------------------
    init_vec = torch.randn(HIDDEN)
    trained_vec = torch.randn(HIDDEN)

    old_model = copy.deepcopy(m)                        # untie, then write: the old path
    _untie_output_embeddings(old_model)
    with torch.no_grad():
        old_model.get_input_embeddings().weight[NEW_ID] = trained_vec

    new_model = copy.deepcopy(m)                        # wrapper: the new path
    emb = NewTokenEmbedding(new_model.get_input_embeddings(), NEW_ID, init_vec)
    new_model.set_input_embeddings(emb)
    with torch.no_grad():
        emb.new_vec.copy_(trained_vec)

    with torch.no_grad():
        ok &= report("NewTokenEmbedding == untie + write row",
                     new_logps(new_model, ids, attn, mask), old_logps(old_model, ids, attn, mask))

    # --- 3. reference_mode == a separate reference model -----------------------
    ref_model = copy.deepcopy(m)
    _untie_output_embeddings(ref_model)
    with torch.no_grad():
        ref_model.get_input_embeddings().weight[NEW_ID] = init_vec

    with torch.no_grad(), emb.reference_mode():
        ok &= report("reference_mode == separate ref model",
                     new_logps(new_model, ids, attn, mask), old_logps(ref_model, ids, attn, mask))

    # --- 4. gradient reaches the vector and nothing else ------------------------
    new_model.train()
    for p in new_model.parameters():
        p.requires_grad = False
    emb.new_vec.requires_grad = True
    new_logps(new_model, ids, attn, mask).sum().backward()

    g = emb.new_vec.grad
    grad_ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"  {'PASS' if grad_ok else 'FAIL'}  {'gradient flows into new_vec':52s} "
          f"norm = {g.norm().item():.3e}" if g is not None else "  FAIL  no gradient")
    leaked = [n for n, p in new_model.named_parameters()
              if p.grad is not None and n.endswith("new_vec") is False]
    print(f"  {'PASS' if not leaked else 'FAIL'}  {'no gradient on frozen parameters':52s} "
          f"{len(leaked)} leaked")
    ok &= bool(grad_ok) and not leaked

    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
