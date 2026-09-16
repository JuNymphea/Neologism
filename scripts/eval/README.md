# LM-judge evaluation

Scores a generation file. Each row of the input `.jsonl` is one question with
two answers — the model's normal answer and its answer under the concept — and
each answer is judged three ways, 0–2 each:

| | question asked of the judge |
|---|---|
| concept relevance | is the concept present in this fragment? |
| instruction relevance | is the fragment on the topic of the instruction? |
| fluency | does it read naturally? |

The three are combined by **harmonic mean, with a zero on any one forcing the
total to zero** — so a fluent answer that ignores the concept scores 0, not 1.3.
Results come back grouped by `factor`, and the `concept` vs `normal` gap is
what the numbers are for.

## Running

```bash
export OPENAI_API_KEY=...        # or GEMINI_API_KEY for a gemini judge

# the training data: how well the target answers fit each concept
python scripts/eval/evaluate.py --source train --input-file a_1 --limit 100
python scripts/eval/evaluate.py --source train --all --limit 100

# what the trained vectors produced (the eval sweep's output)
python scripts/eval/evaluate.py --source results --input-file gemma_a_1_unbiased
python scripts/eval/evaluate.py --source results --input-file 'gemma_*_mixed'
python scripts/eval/evaluate.py --source results --all
```

| source | reads | writes |
|---|---|---|
| `train` | `scripts/train/data/train/*.jsonl` | `scripts/eval/scores/train/` |
| `results` | `scripts/train/results/*.jsonl` | `scripts/eval/scores/results/` |

Each input gets `<name>.json` with the judge's metrics plus a `meta` block
(source, concept, its description, template, judge, question count), and every
run rebuilds `summary.csv` in that folder from all its score files: one row per
file, normal and concept means side by side, and `gap_lm_judge` /
`gap_concept` = concept − normal.

| flag | default | |
|---|---|---|
| `--source` | — | `train` or `results`; where names and patterns are looked up |
| `--input-file` | — | paths, bare names, or glob patterns; several allowed |
| `--all` | off | every `.jsonl` in the source folder |
| `--model` | `gpt-5.2-2025-12-11` | judge; see `SUPPORTED_MODELS` |
| `--limit` | all | judge only the first N questions (2N answers) |
| `--metrics` | all three | `concept` asks only for concept relevance: a third of the calls, and `lm_judge` is then that rating |
| `--overwrite` | off | re-judge even when a score with the same judge and question count exists |
| `--concepts` | `scripts/FrameNet/pos - FrameNet.csv` | `pos - FrameNet.zh.csv` for Chinese results |
| `--scores-dir` | `scripts/eval/scores` | `--out` overrides it for a single file |
| `--cache-dir` | `$JUDGE_CACHE_DIR`, else `scripts/eval/cached_data` | |

**Cost.** Each question is two answers judged three ways, so six calls. A
training file has 2100 questions — 12,600 calls — while a sweep result has 100.
`--limit 100` on the training data makes the two directly comparable at a
fraction of the cost.

The concept id is the first `a_N` / `n_N` / `v_N` field in the filename that
is in the CSV: `a_1.jsonl`, `n_16_gemma_a25.jsonl` → `n_16`,
`gemma_a_1_unbiased.jsonl` → `a_1` with template `unbiased`. (It used to be
the first two fields, which read the sweep's files as `gemma_a`.) Its `gpt`
column is what the judge is shown.

A file whose score already exists with the same judge and question count is
skipped, so a batch can be rerun or resumed freely.

## Caching

Replies are cached on disk keyed by **prompt text alone**, in
`{cache-dir}/persist_lm_cache/{model}_{tag}_cache.pkl`. The caches already paid
for live in `neologism/cached_data/`; reuse them with
`export JUDGE_CACHE_DIR=/Users/shaoshao/Desktop/neologism/neologism/cached_data`. Re-running a file
therefore costs nothing, and re-judging with the same judge reuses everything
already paid for — there are a few hundred MB of those caches. The key scheme
and filenames are unchanged from the original `evaluator/`, so existing caches
still hit.

## What changed from `evaluator/`

The judging itself is **byte-identical**: prompt templates, rating parsing, the
harmonic mean, and the cache keys. Scores stay comparable with everything
already run. `lm_judge.py` differs from the original only in its imports and
one comment — verified by AST comparison. Only the plumbing changed:

- **API keys come from the environment.** They were literals in `evaluate.py`,
  meaning the key sat in every copy of the repo. → `OPENAI_API_KEY` /
  `GEMINI_API_KEY`, with a clear error when unset.
- **`Connection: close` is gone.** With it, each of the 16 concurrent requests
  opened a fresh TCP connection and DNS lookup; on macOS that exhausts the
  resolver and fails as `nodename nor servname provided`.
- **`close()` is allowed to fail.** `chat_completions()` runs `asyncio.run`
  per batch, so the client's loop is already gone by tidy-up time. The
  resulting `Event loop is closed` looked like a failed run, but the scores
  had already been written.
- **Paths are arguments, and missing ones fail loudly.** The concept CSV was
  read at import time from a fixed relative path; a miss left the lookup table
  silently `None` and died much later as a `TypeError`.
- **The model list is data, not control flow.** It was an `if/elif` chain of
  `pass` statements, so each new judge needed an edit mid-function.
- **The CSV header no longer counts as a concept.** It was landing in the map
  as `ID` → `gpt`, which is harmless for lookups but made the count 91.
- `--limit`, a printed summary table, and `mkdir -p` on the output directory.

## Files

```
evaluate.py          driver: load, transform, judge, write
lm_judge.py          the three rubrics and the aggregation
language_models.py   async client + prompt-level disk cache
prompt_templates.py  the three judge prompts (verbatim)
```

The concept CSV (concept id -> description) lives in `scripts/FrameNet/`.

Imports are flat, so run it as a script (`python scripts/eval/evaluate.py`)
from anywhere — Python puts the script's own directory on the path.
