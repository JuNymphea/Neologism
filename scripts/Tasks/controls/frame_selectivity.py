#!/usr/bin/env python
"""Which part of speech does each training frame ask for, in each model?

The template leaves as much of a trace in the vectors of gemma and aya as in qwen's,
but only qwen's is read as the frame's part of speech. One explanation is that the
frames themselves are not equally selective in the three models: "Please X your
answer." may call for a verb in qwen and for a property word in gemma. Nothing is
trained here. Real words of known category are put into the frames exactly as the
token was during training -- raw prompt, a training question in front, no chat
template -- and two things are read:

  expectation  log P(word | left part of the frame): what the model expects there
  licensing    mean surprisal of the right part of the frame after the word (as
               in Task 2): which words the rest of the frame accepts

A frame's preference is read against the unbiased frame, which asks for no
category, so the words' own frequencies cancel.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
from train_neologism import LANGUAGES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", choices=("en", "zh"), required=True)
    ap.add_argument("--words", nargs="+", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("scripts/train/data"))
    ap.add_argument("--n-questions", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    templates = LANGUAGES[args.lang]["templates"]
    qs = []
    for f in sorted((args.data_dir / "train" / args.lang).glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            q = json.loads(line)["question"]
            if q not in qs:
                qs.append(q)
    random.Random(0).shuffle(qs)
    qs = qs[: args.n_questions]

    words = {}
    for p in args.words:
        for pos, ws in json.loads(p.read_text())["words"].items():
            words.setdefault(pos, [])
            words[pos] += [w for w in ws if w not in words[pos]]
    sep = " " if args.lang == "en" else ""
    items = []
    for pos, ws in words.items():
        for w in ws:
            if args.lang == "en" and w[0].lower() in "aeiou":
                continue      # "a {X}" in the noun frame: keep the article right for every word
            ids = tok(sep + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                items.append((pos, w, ids[0]))
    print(f"{len(items)} single-token words: " + str({p: sum(1 for x in items if x[0] == p) for p in words}), flush=True)

    res = {}
    for tname, tmpl in templates.items():
        left_t, right_t = tmpl.split("{token}")
        for qi, q in enumerate(qs):
            left_text = left_t.format(question=q)
            if sep:
                left_text = left_text.rstrip(" ")
            left = tok(left_text, add_special_tokens=True)["input_ids"]
            cont = tok(right_t, add_special_tokens=False)["input_ids"]
            for b in range(0, len(items), args.batch_size):
                chunk = items[b:b + args.batch_size]
                seqs = [left + [wid] + cont for _, _, wid in chunk]
                ids = torch.full((len(seqs), max(map(len, seqs))), pad, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, x in enumerate(seqs):
                    ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
                with torch.no_grad():
                    lg = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float().log_softmax(-1)
                n = len(left)
                for i, (pos, w, wid) in enumerate(chunk):
                    exp = lg[i, n - 1, wid].item()
                    lic = -np.mean([lg[i, t - 1, seqs[i][t]].item() for t in range(n + 1, n + 1 + len(cont))])
                    r = res.setdefault(tname, {}).setdefault(w, {"pos": pos, "exp": [], "lic": []})
                    r["exp"].append(exp); r["lic"].append(lic)
        print(f"  {tname} done", flush=True)
    out = {t: {w: {"pos": r["pos"], "exp": float(np.mean(r["exp"])), "lic": float(np.mean(r["lic"]))}
               for w, r in d.items()} for t, d in res.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "lang": args.lang, "questions": qs,
                                    "templates": templates, "words": out}, ensure_ascii=False, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
