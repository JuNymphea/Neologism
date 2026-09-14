# Task 1 — Explicit POS Probability

Ask the model outright and read the answer off its probabilities:

```
The part of speech of the word "WORD" is ___
```

    P(category | word) = sum over that category's surface forms of P(form | prompt)

normalized across the three categories; the argmax is the prediction.

## What differs from Task 2

Task 1 needs no calibration. There is no reference distribution to fit and no
slot to select, so the model's probabilities *are* the measurement. That also
means it has no place to hide a leak — but it measures something narrower:
metalinguistic knowledge ("is this word called a noun") rather than syntactic
behaviour ("does this word make the continuation predictable").

What it shares with Task 2 is the evidence discipline. Accuracy is reported on
the **same Task 2 `final-test` words**, which neither task was tuned on, so the
two probes are comparable item for item on the same 234 words.

## The one design choice: which surface forms count

Two criteria, and they pull against each other.

**Cover what the model actually writes** — not "the same number of spellings
per category", which looks fair and is not. Dropping `" adj"` throws away real
adjective mass, because a model asked for a part of speech says "adj" quite
naturally. Adding `" n"` and `" v"` to even the count would be much worse: they
are single ambiguous tokens that collect mass from everything else those
letters can begin, inflating noun and verb for reasons unrelated to the
question. So `" adj"` is in and `" n"`/`" v"` are out.

**Keep the forms length-matched across categories.** A form's probability is a
product over its tokens, so a longer spelling is intrinsically smaller, and a
category whose spellings tokenize longer gets penalised for nothing to do with
the word. Under the Gemma tokenizer:

| label | lower | title | upper |
|---|---|---|---|
| noun | `" noun"` **1** | `" Noun"` 2 (`▁N`+`oun`) | `" NOUN"` 2 |
| verb | `" verb"` **1** | `" Verb"` **1** | `" VERB"` 2 |
| adj | `" adjective"` **1** | `" Adjective"` 2 | `" ADJECTIVE"` **3** |
| adj | `" adj"` **1** | `" Adj"` **1** | `" ADJ"` 2 |

The casings are uneven in *opposite* directions: title case hands verb a
one-token spelling where noun gets two, while upper case costs adjective three
tokens where noun pays two. Neither slant is principled — it is just how the
vocabulary happens to have been carved.

Articles, by contrast, are length-neutral: exactly one extra token for every
category. Hence the default set is lowercase only:

```
noun   " noun"(1)       " a noun"(2)
verb   " verb"(1)       " a verb"(2)
adj    " adjective"(1)  " an adjective"(2)
       " adj"(1)        " an adj"(2)
```

Every bare form is 1 token and every articled form is 2, in all three
categories. Adjective has twice as many entries only because it has two
labels, and each is length-matched to its noun/verb counterpart.

Casing applies to the **whole answer**, never to one part of it: an answer is
`" a noun"`, `" A Noun"` or `" A NOUN"`, never `" A noun"`. Casing the article
independently of the label is what produced an earlier version in which
`" A noun"` was covered but `" Noun"` was not.

## Sensitivity

`--report-sensitivity` reports the alternatives side by side, each isolating
one decision:

| set | flag | sizes |
|---|---|---|
| +首字母大写 | `--casings lower,title` | 4/4/8, tokens 1–3 |
| +全大写 | `--casings lower,upper` | 4/4/8, tokens 1–4 |
| 三种大小写全上 | `--casings lower,title,upper` | 6/6/12 |
| 无缩写 | `--no-abbrev` | 2/2/2 |
| 无冠词 | `--no-articles` | 1/1/2 |
| 最小集 | both | 1/1/1, all single-token |

The two casing levels are listed separately because their length bias runs in
opposite directions — reporting only a combined "with casing" row would let
them cancel and hide both.

It costs no extra GPU time: every set draws from the same union of surface
forms, which is scored once, so the report is pure re-aggregation. The run
also saves `log_probabilities`, so any further variant set can be evaluated
later with no model at all.

## Two scripts

**`run_task1.py`** scores one word file. By default that is Task 2's
`control_finaltest.json`, so the headline number is comparable to Task 2's
item for item. `--words` also takes a plain JSON list, which is scored without
an accuracy report — the path for neologisms.

**`eval_on_task2_words.py`** is the fuller evaluation, and does two things the
first does not.

*It uses all three Task 2 splits, not just final-test.* Task 1 fits nothing and
selects nothing, so calibration (148/POS) and probe-dev (74/POS) were never
spent on it and are exactly as clean for Task 1 as final-test is. That is 300
words per category instead of 78 — roughly half the confidence interval. The
splits are reported separately all the same, because only the final-test row is
comparable to Task 2's headline; `combined` is Task 1's own best estimate.

*It compares the two probes item by item.* Equal accuracies do not imply equal
predictions, so the run reports the 2×2 table of who got what right, the
agreement rate, and McNemar's exact test. McNemar is the right test because
both probes are measured on the *same* words: only the discordant pairs carry
information about which probe is better. Task 2's per-word predictions are
rebuilt from its frozen fits in `calibration.json` + `surprisal.json` — the
reconstruction reproduces its reported 0.8205 / 0.923 / 0.667 / 0.872 exactly,
which is the check that it is the same classifier.

```bash
python Task_1/run_task1.py --model google/gemma-3-4b-it --report-sensitivity
python Task_1/eval_on_task2_words.py --model google/gemma-3-4b-it
sbatch scripts/Tasks/Task_1/submit_task1.slurm      # runs both
```

## Output

`out/task1.json` — per-word P(noun), P(verb), P(adj), the variant set used, the
accuracy report with a confusion matrix, the sensitivity table, and the raw
per-form log probabilities.

`out/task1_on_task2_words.json` — the same per split, the combined report, and
`comparison_with_task2` holding the 2×2 counts, agreement, McNemar p, and the
per-word gold/Task 1/Task 2 triples.
