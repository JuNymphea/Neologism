#!/usr/bin/env python
"""Induce Task 2 syntactic slots from UD English EWT.

    python Task_2/run_induction.py

Pipeline (see slotgen/ for the individual stages):

    1-2  every NOUN/VERB/ADJ token -> hybrid context signature   (frames.py)
    3    N(c, p) and P(p | c) over the treebank                  (induce.py)
    4    frequency / lexical-productivity / purity thresholds     (induce.py)
    5    rank by frequent x productive x selective                (induce.py)
    6    de-duplicate by construction family                      (induce.py)
    7    realize each frame as a minimal probe item               (minimal.py)
    +    re-estimate purity on held-out EWT and on UD-GUM         (validate.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slotgen.minimal import (  # noqa: E402
    TRAINING_TEMPLATES,
    collides_with_training,
    make_item,
)
from slotgen.tagging import TaggerCheck, load_tagger  # noqa: E402
from slotgen.frames import TARGET_POS, SignatureEncoder  # noqa: E402
from slotgen.overrides import Overrides, write_review_template  # noqa: E402
from slotgen.induce import (  # noqa: E402
    DevFilter,
    StructuralConstraints,
    count_signatures,
    induce,
)
from slotgen.ud_io import (  # noqa: E402
    bigram_counts,
    form_distributions,
    frequent_adverbs,
    load_corpus,
)
from slotgen.validate import recount, summarize, validate  # noqa: E402

POS_LABEL = {"NOUN": "Noun", "VERB": "Verb", "ADJ": "Adjective"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ud-dir", type=Path, default=here / "data" / "ud")
    p.add_argument("--out-dir", type=Path, default=here / "out")
    p.add_argument("--n-slots", type=int, default=10,
                   help="slots to select per part of speech")
    p.add_argument("--min-purity", type=float, default=0.90,
                   help="minimum P(p|c); never relaxed. With the all-UPOS "
                        "denominator 0.90 is stricter than 0.95 three-way.")
    p.add_argument("--purity-denominator", choices=("all", "target3"),
                   default="all",
                   help="all: P(p|c) over every UPOS filling the context "
                        "(stricter); target3: over NOUN/VERB/ADJ only")
    p.add_argument("--max-left", type=int, default=2)
    p.add_argument("--min-right", type=int, default=1,
                   help="frames must have right context: Task 2 scores the "
                        "continuation, which would otherwise be unattested")
    p.add_argument("--max-right", type=int, default=3)
    p.add_argument("--n-continuation", type=int, default=4,
                   help="continuation length the surprisal probe will score; "
                        "every slot is guaranteed this many non-punctuation "
                        "tokens after the neologism")
    p.add_argument("--min-bigram", type=int, default=3,
                   help="a bigram of added material must be attested at least "
                        "this often in the pooled treebanks")
    p.add_argument("--adv-topk", type=int, default=30,
                   help="how many ADV forms to keep lexicalized")
    p.add_argument("--prune-below", type=int, default=5,
                   help="drop signatures rarer than this before collecting stats")
    p.add_argument("--min-left", type=int, default=1,
                   help="a slot must be a two-sided frame, not a one-sided context")
    p.add_argument("--min-content-right", type=int, default=1,
                   help="right context must contain non-punctuation to score on")
    p.add_argument("--min-lexical-cues", type=int, default=1,
                   help="a slot must contain at least one grammatical cue word")
    p.add_argument("--dev-min-count", type=int, default=8,
                   help="dev evidence needed before a frame can be rejected")
    p.add_argument("--dev-min-purity", type=float, default=0.85,
                   help="selectivity a frame must retain on the dev corpus")
    p.add_argument("--probe-words", type=Path,
                   default=here.parent / "pos_words_sampled.json",
                   help="known-POS words used to verify probe items")
    p.add_argument("--min-tagger-accuracy", type=float, default=0.85,
                   help="tagger agreement a probe item must reach")
    p.add_argument("--overrides", type=Path, default=here / "realizations.json",
                   help="human review file: accept / rewrite / reject per frame")
    p.add_argument("--write-review", action="store_true",
                   help="(re)write the review file from the current output")
    p.add_argument("--allow-unrealizable", action="store_true",
                   help="keep frames that cannot be realized as a minimal "
                        "probe item (default: drop and take the next candidate)")
    return p.parse_args()


def emit_markdown(selected, realized, thresholds, summary, validation, args) -> str:
    lines: list[str] = []
    lines.append("# Corpus-derived syntactic slots for Task 2\n")
    lines.append(
        f"Induced from UD English EWT (train split), {args.n_slots} slots per POS. "
        f"Purity floor P(p|c) >= {args.min_purity:.2f}, never relaxed; frequency and "
        "lexical-productivity thresholds relaxed only as needed.\n"
    )

    lines.append("\n## Thresholds actually used\n")
    lines.append("| POS | thresholds | candidates selected |")
    lines.append("|---|---|---|")
    for pos in TARGET_POS:
        lines.append(
            f"| {POS_LABEL[pos]} | {thresholds[pos].describe()} | {len(selected[pos])} |"
        )

    lines.append("\n## Selectivity, held out and across corpora\n")
    lines.append(
        "Mean P(p|c) over the selected slots. `train` is the split the slots were "
        "chosen on; the rest were never used for selection.\n"
    )
    cols = list(summary["per_pos"][TARGET_POS[0]].keys())
    lines.append("| POS | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for pos in TARGET_POS:
        row = summary["per_pos"][pos]
        lines.append(
            f"| {POS_LABEL[pos]} | "
            + " | ".join("-" if row[c] is None else str(row[c]) for c in cols)
            + " |"
        )

    if summary["flagged"]:
        lines.append("\n**Frames whose selectivity did not hold up:**\n")
        lines.append("| POS | signature | corpus | train | held-out | n |")
        lines.append("|---|---|---|---|---|---|")
        for f in summary["flagged"]:
            lines.append(
                f"| {POS_LABEL[f['pos']]} | `{f['signature']}` | {f['corpus']} | "
                f"{f['train_purity']} | {f['heldout_purity']} | {f['heldout_n']} |"
            )
    else:
        lines.append(
            "\nNo selected frame fell below the held-out selectivity floor.\n"
        )

    lines.append("\n## Slots\n")
    for pos in TARGET_POS:
        lines.append(f"\n### {POS_LABEL[pos]}\n")
        lines.append(
            "| # | probe item | scored `D_f` | signature | P(p\\|c) | N | types |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(realized[pos], 1):
            it = r.get("probe_item")
            if not it:
                lines.append(f"| {i} | _(none)_ | | `{r['signature']}` | | | |")
                continue
            lines.append(
                f"| {i} | {it['text']} | `{' '.join(it['diagnostic'])}` | "
                f"`{r['signature']}` | {r['purity']:.2f} | {r['n_target']} | "
                f"{r['n_types']} |"
            )
    return "\n".join(lines) + "\n"


def emit_latex(realized, use: str = "probe_item") -> str:
    """Table 5 replacement."""
    rows = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{POS} & \textbf{Slot} & $P(p\mid c)$ & $N$ \\",
        r"\midrule",
    ]
    for pos in TARGET_POS:
        first = True
        for r in realized[pos]:
            real = r[use] or r["realization_template"] or r["realization_attested"]
            if real is None:
                continue
            text = real["text"].replace("{NEOLOGISM}", r"\texttt{NEOLOGISM}")
            text = text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            label = POS_LABEL[pos] if first else ""
            rows.append(f"{label} & {text} & {r['purity']:.2f} & {r['n_target']} \\\\")
            first = False
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows += [
        r"\end{tabular}",
        r"\caption{Syntactic slots for Task 2, induced from UD English EWT by "
        r"frequency, lexical productivity and category selectivity. "
        r"$P(p\mid c)$ and $N$ are estimated on the training split.}",
        r"\label{tab:slots}",
        r"\end{table}",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Three disjoint roles. Frames are discovered on EWT train; the dev pool can
    # veto a frame whose selectivity does not replicate; the test pool is only
    # ever read to report numbers, so nothing in the reported figures was used
    # to choose the slots.
    ewt_train = args.ud_dir / "en_ewt-ud-train.conllu"
    # GUM is kept entirely out of selection so that "cross-corpus
    # generalization" is a claim about data that played no part in any decision.
    dev_files = [args.ud_dir / "en_ewt-ud-dev.conllu"]
    test_files = {
        "ewt_test": [args.ud_dir / "en_ewt-ud-test.conllu"],
        "gum_external": sorted(args.ud_dir.glob("en_gum-ud-*.conllu")),
    }
    dev_files = [p for p in dev_files if p.exists()]
    test_files = {k: [p for p in v if p.exists()] for k, v in test_files.items()}
    test_files = {k: v for k, v in test_files.items() if v}

    print(f"[1/6] loading {ewt_train.name}")
    train = load_corpus([ewt_train])
    n_tokens = sum(len(s) for s in train)
    print(f"      {len(train):,} sentences, {n_tokens:,} tokens")

    adverbs = frequent_adverbs(train, args.adv_topk)
    encoder = SignatureEncoder(lexicalized_adverbs=adverbs)

    print(f"[2/6] extracting context signatures "
          f"(left<={args.max_left}, right {args.min_right}-{args.max_right})")
    frames = count_signatures(
        train, encoder,
        max_left=args.max_left, min_right=args.min_right,
        max_right=args.max_right, prune_below=args.prune_below,
        purity_denominator=args.purity_denominator,
    )
    print(f"      {len(frames):,} signatures occurring >={args.prune_below} times")

    print("[3/6] thresholding, scoring and de-duplicating by construction")
    dev = load_corpus(dev_files)
    all_files = [ewt_train, *dev_files, *[p for v in test_files.values() for p in v]]
    # Corpus-internal naturalness signals: which syntactic positions a filler
    # actually occurs in, and which adjacent word pairs are attested at all.
    _pooled = load_corpus(all_files)
    dists = form_distributions(_pooled)
    bigrams = bigram_counts(_pooled)
    del _pooled
    print(f"      {len(dists):,} word forms with attested (UPOS, deprel); "
          f"{len(bigrams):,} distinct bigrams")
    probes = json.loads(args.probe_words.read_text())
    probes = {k.upper().replace("ADJ", "ADJ"): v for k, v in probes.items()}
    check = TaggerCheck(load_tagger(), probes)
    overrides = Overrides.load(args.overrides)
    if overrides.entries:
        print(f"      review file: {len(overrides.entries)} frames "
              f"({len(overrides.rejected())} rejected by hand)")
    _items: dict = {}
    collisions: dict = {}

    def item_for(fs):
        """The minimal probe item for a frame, cached. None => unusable."""
        sig = fs.signature
        if overrides.status(sig) == "reject":
            return None
        # A frame that duplicates a training template would measure memory of
        # that construction rather than lexical-category generalization.
        clash = collides_with_training(sig, encoder)
        if clash is not None:
            collisions[sig] = clash
            return None
        if sig not in _items:
            _items[sig] = make_item(
                fs, encoder, check, min_accuracy=args.min_tagger_accuracy,
                dists=dists, bigrams=bigrams, min_bigram=args.min_bigram,
            )
        return _items[sig]

    def probe_item(fs):
        """The item as a dict, honouring a hand-written override if present."""
        rewritten = overrides.rewritten(fs.signature, 1)
        if rewritten is not None:
            return rewritten
        item = item_for(fs)
        return item.as_dict() if item is not None else None

    def realizable(fs) -> bool:
        """A frame is usable only if it yields a minimal probe item.

        Checked during selection, so an unrealizable frame is replaced by the
        next-best corpus candidate rather than leaving a hole in the inventory.
        """
        if args.allow_unrealizable:
            return True
        return probe_item(fs) is not None

    print(f"      dev pool: {len(dev):,} sentences "
          f"({', '.join(p.name for p in dev_files)})")
    dev_filter = DevFilter(
        counts=recount(dev, encoder, frames.keys(),
                       args.max_left, args.min_right, args.max_right),
        min_count=args.dev_min_count,
        min_purity=args.dev_min_purity,
    )
    constraints = StructuralConstraints(
        encoder=encoder,
        min_left=args.min_left,
        min_content_right=args.min_content_right,
        min_lexical_cues=args.min_lexical_cues,
    )
    def item_fingerprint(fs):
        """The realized item, so two signatures cannot yield the same test."""
        it = item_for(fs)
        return None if it is None else it.text

    selected, thresholds = induce(
        frames, n_slots=args.n_slots, min_purity=args.min_purity,
        constraints=constraints, dev_filter=dev_filter, validator=realizable,
        fingerprint=item_fingerprint,
    )
    for pos in TARGET_POS:
        print(f"      {POS_LABEL[pos]:<10} {len(selected[pos])} slots "
              f"({thresholds[pos].describe()})")

    print("[4/6] building minimal probe items")
    realized = {
        pos: [{**fs.to_dict(), "probe_item": probe_item(fs)} for fs in frames_]
        for pos, frames_ in selected.items()
    }

    print("[5/6] re-estimating selectivity on the untouched test corpora")
    corpora = {name: load_corpus(paths) for name, paths in test_files.items()}
    validation = validate(
        selected, encoder, corpora,
        max_left=args.max_left, min_right=args.min_right, max_right=args.max_right,
    )
    summary = summarize(selected, validation)
    for pos in TARGET_POS:
        print(f"      {POS_LABEL[pos]:<10} {summary['per_pos'][pos]}")

    print("[6/6] writing outputs")
    sig_to_frame = {fs.signature: fs for fr in selected.values() for fs in fr}
    for pos in TARGET_POS:
        for r in realized[pos]:
            dev_purity = dev_filter.purity_of(sig_to_frame[r["signature"]])
            r["validation"] = {
                "dev": None if dev_purity is None else round(dev_purity, 4),
                **{
                    name: est[r["signature"]].to_dict()
                    for name, est in validation.items()
                },
            }

    (args.out_dir / "slots.json").write_text(
        json.dumps(
            {
                "config": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "corpus": {
                    "induction": ewt_train.name,
                    "n_sentences": len(train),
                    "n_tokens": n_tokens,
                    "dev_pool": [p.name for p in dev_files],
                    "test_pools": {k: [p.name for p in v]
                                   for k, v in test_files.items()},
                    "lexicalized_adverbs": sorted(adverbs),
                },
                "thresholds": {p: vars(thresholds[p]) for p in TARGET_POS},
                "validation_summary": summary,
                "training_templates": TRAINING_TEMPLATES,
                "construction_collisions": collisions,
                "slots": realized,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (args.out_dir / "frames_all.jsonl").open("w", encoding="utf-8") as fh:
        ranked = sorted(frames.values(), key=lambda f: f.score, reverse=True)
        for fs in ranked:
            if fs.n_target >= 5:
                fh.write(json.dumps(fs.to_dict(), ensure_ascii=False) + "\n")

    # A minimal file for the surprisal probe to consume: the slot sentence and
    # the continuation tokens whose surprisal Task 2 scores.
    flat = {
        pos: [
            {
                "signature": r["signature"],
                # The string the model is conditioned on. Substitute the
                # neologism (or a control word) for {NEOLOGISM}.
                "prefix": " ".join(r["probe_item"]["prefix_tokens"]),
                "prefix_tokens": r["probe_item"]["prefix_tokens"],
                "morph_requirement": r["probe_item"].get("morph_requirement"),
                # D_f: the only tokens whose surprisal is scored.
                "diagnostic": r["probe_item"]["diagnostic"],
                "item": r["probe_item"]["text"],
                "purity": r["purity"],
                "n": r["n_target"],
            }
            for r in realized[pos]
            if r["probe_item"]
        ]
        for pos in TARGET_POS
    }
    (args.out_dir / "slots_flat.json").write_text(
        json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (args.out_dir / "slots.md").write_text(
        emit_markdown(selected, realized, thresholds, summary, validation, args),
        encoding="utf-8",
    )
    (args.out_dir / "slots_table.tex").write_text(
        emit_latex(realized), encoding="utf-8"
    )
    print(f"      -> {args.out_dir}/slots.json")
    print(f"      -> {args.out_dir}/slots_flat.json")
    print(f"      -> {args.out_dir}/slots.md")
    print(f"      -> {args.out_dir}/slots_table.tex")
    print(f"      -> {args.out_dir}/frames_all.jsonl")

    review = args.out_dir / "review_template.json"
    write_review_template(review, realized)
    print(f"      -> {review}  (copy to {args.overrides.name} and edit to "
          f"accept / rewrite / reject)")


if __name__ == "__main__":
    main()
