# Task 2 — Corpus-Calibrated Syntactic Compatibility Probe

Task 2 (§3.3.2) asks one question:

> When a neologism is placed in a local syntactic environment that, in real
> text, strongly selects one part of speech, does its continuation behaviour
> look more like known words of that POS than of the other two?

The 30 slots in the paper's Table 5 were written by hand, so the probe rested
on the authors' intuitions about which sentences are noun-, verb- or
adjective-selective. This directory replaces them with frames **induced from a
POS-annotated treebank**, realized as **minimal probe items**, and calibrated
against **known single-token words**.

## Five components, five separate questions

| Component | Question it answers | Where |
|---|---|---|
| Corpus induction | Which local environments actually select a POS? | `slotgen/induce.py` |
| Minimal realization | How to present one without concept-specific semantics? | `slotgen/minimal.py` |
| Diagnostic continuation | What does this construction really require the model to predict? | `D_f`, below |
| Known-word calibration | What surprisal counts as "noun-like"? | `calibrate_probe.py` |
| Neologism | Which known category does it behave like? | (not in this directory) |

## Pipeline

```
run_induction.py        corpus  ->  30 probe items
build_control_words.py  vocabulary  ->  200 calibration + 100 validation words per POS
score_probe.py          model  ->  S(w, f) for every word x slot        [needs GPU]
calibrate_probe.py      fit, diagnose per slot, report accuracy
```

The expensive forward passes happen once, in `score_probe.py`, and are written
to `out/surprisal.json`. Every later decision -- thresholds, slot rejection,
plots -- is re-derivable from that file without a GPU.

## Probe items are minimal, and only the licensed continuation is scored

A probe item is a minimal grammatical **prefix**, plus the continuation the
frame itself licenses. Nothing is padded to reach a sentence boundary or a
fixed token count.

```
corpus frame          probe item                          scored D_f
---------------------------------------------------------------------
the {SLOT} of         The {NEOLOGISM} of                  of
to {SLOT} PRON        They tried to {NEOLOGISM} it        it
very {SLOT} NOUN      It was a very {NEOLOGISM} thing     thing
the {SLOT} and NOUN   The {NEOLOGISM} and something       and something
```

    S(w, f) = (1 / |D_f|) * sum_{t in D_f} -log P(t | prefix containing w)

`|D_f|` follows the construction rather than a fixed n. A fixed window has no
theoretical justification and mixes in tokens that carry no evidence about the
slot. An earlier version padded every item to four scored tokens with tails
like *at the time* / *more than once*; those tokens repeated across slots and
diluted the measurement.

## Induction

Signatures are hybrid: grammatical material verbatim, open-class material as a
UPOS placeholder.

```
She decided to leave it .    (leave/VERB)    ->  to {SLOT} PRON
The problem was serious .    (problem/NOUN)  ->  the {SLOT} was
```

`score(c, p) = log(1 + N_c) * P(p|c) * (1 - H_norm(c)) * log(1 + T_c)`

**`P(p|c)` is computed over all UPOS**, not just the three target categories.
That matters: `as {SLOT} as` is 100% adjective among noun/verb/adjective, but
adjectives are only 28% of what actually fills that context -- the rest are
adverbs (*as quickly as*), a category not in the label set. The threshold is
0.90 all-UPOS, which is stricter than the 0.95 three-way figure it replaced.

Frequency and lexical-productivity thresholds relax down a ladder when a
category cannot fill ten slots; **purity never relaxes**.

Frames are also required to be **construction-disjoint from the training
templates** (`slotgen/minimal.py`). Scoring a neologism in the construction it
was trained in would measure memory of that construction rather than
generalization about lexical category.

## Corpus roles

| Role | Data | Used for |
|---|---|---|
| Induction | EWT train (12.5k sents) | discovering and ranking frames |
| Dev | EWT dev | vetoing frames whose selectivity does not replicate |
| Test | EWT test | reporting only |
| **External** | **all of UD-GUM** | **never touched during selection** |

## Control words

`build_control_words.py` produces two disjoint files, so that neither can be
reached for by accident:

| File | Size | Role |
|---|---|---|
| `out/control_calibration.json` | 200/POS | fit the reference distributions |
| `out/control_validation.json` | 100/POS | report accuracy, drop slots -- never fitted on |

Four things have to be right, and each one silently invalidates the
calibration if it is not:

1. **Single token in the position the probe uses.** `" word"`, not `"word"`.
   The earlier lists tested vocabulary membership instead, letting through
   75/300 nouns that are two tokens after a space (`" abbot"` -> `_ab` + `bot`,
   because only `_Abbot` is in the vocabulary) -- mostly proper nouns.
2. **Unambiguous POS**, so `plate`, `run`, `dark` are out.
3. **The inflection the slot requires.** All thirty slots want a bare form, and
   the script verifies this rather than assuming it. Five slots read
   `A {SLOT} ...`, so mass nouns are removed using corpus evidence (a noun seen
   often but never with "a"/"an").
4. **Matched frequency across categories.** A frequent word has lower
   continuation surprisal whatever its category. Stratified sampling over
   token-id rank gives KS p > 0.97 for all three pairs.

## Calibration and slot rejection

Per slot, fit matched and non-matched Gaussians on the calibration words, then

    r_i(w) = log p(s_i(w) | match) - log p(s_i(w) | non-match)

Per-slot AUC and three-way accuracy are reported on the **validation** words.
A slot that cannot separate matched from non-matched words is a probe failure
and is dropped.

> That decision is made **only** from known control words, before any neologism
> is scored. Choosing slots by their effect on the main result would be probe
> tuning; the ordering in `calibrate_probe.py` is what prevents it.

Corpus purity and control-word AUC are two different validities: the first says
the environment is POS-selective in real language, the second says the model
can feel the difference.

## Human review

`out/review_template.json` is rewritten on every run. Copy it to
`realizations.json` and mark each item `accept` / `rewrite` / `reject`; a
rejected frame is dropped and the next corpus candidate takes its place. This
does not compromise the method -- a human never decides *which* POS an
environment selects, only whether a given item is usable English.

## Running

```bash
conda activate py310

python Task_2/run_induction.py
python Task_2/build_control_words.py
python Task_2/score_probe.py --model google/gemma-3-4b-it --batch-size 64
python Task_2/calibrate_probe.py --min-auc 0.60
```

Treebanks are in `data/ud/`. To refresh:

```bash
for f in train dev test; do
  curl -sL -o Task_2/data/ud/en_ewt-ud-$f.conllu \
    https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-$f.conllu
  curl -sL -o Task_2/data/ud/en_gum-ud-$f.conllu \
    https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-$f.conllu
done
```

The word lists depend on the tokenizer. If the model changes, regenerate
`pos_words_v2.json` with `scripts/framenet/extract_pos_words.py` and re-run
`build_control_words.py`.

## Outputs (`out/`)

| File | Contents |
|---|---|
| `slots_flat.json` | The 30 items: `prefix_tokens` (with `{NEOLOGISM}`) and `diagnostic` -- what the probe consumes |
| `slots.json` | Everything: statistics, per-corpus validation, construction collisions, config |
| `slots.md` | Human-readable report |
| `slots_table.tex` | Table 5 replacement |
| `frames_all.jsonl` | Every candidate frame with its statistics -- the evidence behind the selection |
| `control_{calibration,validation}.json` | The two word sets |
| `control_words_meta.json` | Filter counts, frequency-match evidence, KS tests |
| `surprisal.json` | S(w, f) matrix (from `score_probe.py`) |
| `calibration.json` | Fits, per-slot AUC, kept/dropped slots, accuracy |

## An incidental finding worth reporting

All ten hand-written adjective slots in Table 5 are **predicative** (`seems`,
`sounds`, `looks`, `is`, `became`, `feels`, `remained`). The induced set is
dominated by **attributive** frames.

This bears on the metric itself. Task 2 scores tokens *after* the slot, so a
slot is diagnostic only when the following material depends on the target's
category. In attributive position a head noun follows; in predicative position
the adjective sits near the end of its clause and little downstream depends on
it. Asked which environments actually select adjectives, the corpus returns
mostly the diagnostic ones -- a candidate explanation for the 54% adjective
accuracy at n=1 in Table 1, and for the non-significant adjective steering in
Section 5.2.

## Approaches tried and abandoned

| Approach | Why it failed |
|---|---|
| Attested corpus sentences | Natural, but they put concept-specific semantics back in |
| Per-position argmax assembly | *"The the {NEOLOGISM} of people and the."* |
| Most-frequent left and right spans, joined | Each half attested, the join not: *"Would a more {NEOLOGISM} experience would be extremely appreciated."* |
| LM perplexity as a naturalness judge | **Circular** -- Task 2 measures LM continuation surprisal, so selecting slots by it favours slots whose continuation is predictable regardless of the filler, which are the *worst* at separating matched from non-matched |
| Padding to a fixed 4-token window | The tail carries no syntactic evidence and repeats across slots |
| One fixed filler per UPOS (ADJ -> other) | Right tag, wrong syntactic distribution: *"very X and other"* |

`REPORT_zh.md` is the Chinese counterpart of this file.
