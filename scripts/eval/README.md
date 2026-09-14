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

python scripts/eval/evaluate.py --input-file n_16_gemma.jsonl
python scripts/eval/evaluate.py --input-file res/x.jsonl \
    --model gemini-2.5-flash --limit 100
```

| flag | default | |
|---|---|---|
| `--input-file` | — | a path, or a bare filename looked up under `--res-dir` |
| `--model` | `gpt-5.2-2025-12-11` | judge; see `SUPPORTED_MODELS` |
| `--limit` | all | judge only the first N questions (2N answers) |
| `--concepts` | `scripts/framenet/pos - FrameNet.csv`, else the copy here | |
| `--res-dir` | `res/framenet/res` | |
| `--scores-dir` | `scores/framenet` | `--out` overrides with an exact path |
| `--cache-dir` | `cached_data` | |

The concept id is read from the filename: `n_16_gemma_a25.jsonl` → `n_16`,
looked up in the CSV's `gpt` column, and that description is what the judge is
shown.

## Caching

Replies are cached on disk keyed by **prompt text alone**, in
`{cache-dir}/persist_lm_cache/{model}_{tag}_cache.pkl`. Re-running a file
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
pos - FrameNet.csv   concept id -> description, so this folder runs standalone
```

Imports are flat, so run it as a script (`python scripts/eval/evaluate.py`)
from anywhere — Python puts the script's own directory on the path.
