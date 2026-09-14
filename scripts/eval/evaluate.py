#!/usr/bin/env python
"""Score a generation file with the LM judge.

Each row of the input `.jsonl` holds one question with two answers -- the
model's normal answer and its answer under the concept -- and both are judged
on concept relevance, instruction relevance and fluency, 0-2 each, combined by
harmonic mean. The scores come back grouped by `factor`, so `concept` versus
`normal` is the comparison the numbers are for.

    python scripts/eval/evaluate.py --input-file n_16_gemma.jsonl
    python scripts/eval/evaluate.py --input-file res/x.jsonl --model gemini-2.5-flash --limit 100

The judging itself -- prompts, rating parsing, the harmonic mean, the cache
key scheme -- is byte-identical to the original `evaluator/`. Only the
plumbing around it changed, so scores stay comparable with everything already
run, and the existing caches under `cached_data/persist_lm_cache/` still hit.

What did change, and why:

  - **API keys come from the environment.** They used to be literals in this
    file, which meant the key was in every copy of the repo and in its history.
    Set `OPENAI_API_KEY` / `GEMINI_API_KEY`.
  - **No `Connection: close`.** With it, each of the 16 concurrent requests
    opened a fresh TCP connection and DNS lookup; on macOS that exhausts the
    resolver and fails as `nodename nor servname provided`. Keep-alive fixes it.
  - **`close()` is allowed to fail.** `chat_completions()` runs `asyncio.run`
    per batch, so the loop the client was created on is already closed by the
    time we get here, and the tidy-up raised `Event loop is closed` *after* the
    scores had been written -- which looked like a failed run but was not.
  - **Paths are arguments, and missing ones fail loudly.** The concept CSV was
    read at import time from a fixed relative path; when that missed, the
    lookup table was silently `None` and the run died much later with a
    confusing `TypeError`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

from language_models import LanguageModel
from lm_judge import LMJudgeEvaluator

DEFAULT_MODEL = "gpt-5.2-2025-12-11"
#: Where the judge's replies are cached between runs, keyed by prompt text.
DEFAULT_CACHE_DIR = "cached_data"
DEFAULT_CACHE_TAG = "demo"


CONCEPT_CSV_NAME = "pos - FrameNet.csv"


def repo_root() -> Path:
    """The directory the relative default paths are written against."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts").is_dir() and (parent / "scripts") != here.parent:
            return parent
    return Path.cwd()


def default_concept_csv() -> Path:
    """A copy sits next to this script so the folder runs on its own; the one
    under `scripts/framenet/` wins if it is there, since that is the original.
    """
    here = Path(__file__).resolve().parent
    candidates = [repo_root() / "scripts" / "framenet" / CONCEPT_CSV_NAME,
                  here / CONCEPT_CSV_NAME]
    return next((c for c in candidates if c.exists()), candidates[-1])


def read_concept_map(path: Path) -> dict:
    """{concept id: concept description}, from columns 2 and 4 of the CSV.

    The CSV is `POS,ID,FrameNet,gpt`; the judge is shown the `gpt` column.
    Raises rather than returning None -- an empty map used to surface a long
    way downstream as a `TypeError` on a subscript.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"concept CSV not found: {path}\n"
            f"pass --concepts explicitly, e.g. "
            f"--concepts 'scripts/framenet/{CONCEPT_CSV_NAME}'")
    result = {}
    # utf-8-sig strips the BOM, which would otherwise show up in the first column.
    with open(path, mode="r", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if len(row) < 4:
                continue
            key, value = row[1].strip(), row[3].strip()
            # The header row would otherwise land in the map as "ID" -> "gpt",
            # harmless for lookups but it makes the count wrong (91, not 90).
            if i == 0 and key.upper() == "ID":
                continue
            if key and value:
                result[key] = value
    if not result:
        raise ValueError(f"no usable rows in {path} (expected POS,ID,FrameNet,gpt)")
    return result


def transform_res_data(input_file: Path, concept_map: dict,
                       limit: int | None = None) -> pd.DataFrame:
    """One judged row per answer: each question contributes normal + concept."""
    df = pd.read_json(input_file, lines=True)
    if limit:
        df = df.head(limit)

    # "n_16_gemma_a25.jsonl" -> "n_16". The concept id is the first two
    # underscore-separated fields of the filename.
    parts = Path(input_file).stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"cannot read a concept id from '{Path(input_file).name}'; "
            f"expected a name like 'n_16_....jsonl'")
    concept = f"{parts[0]}_{parts[1]}"
    if concept not in concept_map:
        raise KeyError(
            f"concept '{concept}' (from {Path(input_file).name}) is not in the "
            f"concept CSV. Known ids include: "
            f"{', '.join(sorted(concept_map)[:8])}...")

    rows = []
    for _, row in df.iterrows():
        question = row["question"]
        rows.append({"original_prompt": question,
                     "generation": row["normal_answer"], "factor": "normal"})
        rows.append({"original_prompt": question,
                     "generation": row["concept_answer"], "factor": "concept"})

    new_df = pd.DataFrame(rows)
    new_df["dataset_name"] = concept
    new_df["input_concept"] = concept_map[concept]
    return new_df


def get_chat_client(model_name: str) -> AsyncOpenAI:
    """An async client pointed at whichever provider serves this model."""
    if "gemini" in model_name.lower():
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.environ.get("GEMINI_API_KEY")
        var = "GEMINI_API_KEY"
    else:
        base_url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
        var = "OPENAI_API_KEY"
    if not api_key:
        raise SystemExit(
            f"{var} is not set. Export it before running:\n"
            f"    export {var}=...\n"
            f"(It used to be hard-coded in this file; it is not any more.)")

    # Connections are reused. The previous `Connection: close` header forced a
    # new TCP connection and DNS lookup per request, which the macOS resolver
    # gives up on under concurrency.
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        ),
        max_retries=3,
    )


def evaluate(input_file: Path, scores_file: Path, model: str,
             concept_map: dict, cache_dir: str = DEFAULT_CACHE_DIR,
             cache_tag: str = DEFAULT_CACHE_TAG,
             limit: int | None = None) -> dict:
    lm_model = LanguageModel(
        model=model,
        client=get_chat_client(model),
        use_cache=True,
        cache_level="prompt",
        master_data_dir=cache_dir,
        cache_tag=cache_tag,
    )
    evaluator = LMJudgeEvaluator(lm_model=lm_model, model_name=model)

    scores = evaluator.compute_metrics(
        transform_res_data(input_file, concept_map, limit))

    scores_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    lm_model.save_cache()
    try:
        asyncio.run(lm_model.close())
    except RuntimeError as e:
        # Expected: chat_completions() ran asyncio.run per batch, so the loop
        # the client bound to is long gone. The scores are already on disk.
        print(f"（关闭 HTTP 客户端时的预期报错，已忽略：{e}）")
    return scores


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", "--input_file", dest="input_file", required=True,
                    help="a path, or a filename to look up under --res-dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concepts", type=Path, default=default_concept_csv())
    ap.add_argument("--res-dir", type=Path, default=root / "res" / "framenet" / "res")
    ap.add_argument("--scores-dir", type=Path, default=root / "scores" / "framenet")
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit output path; overrides --scores-dir")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--cache-tag", default=DEFAULT_CACHE_TAG)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N questions (2N answers)")
    args = ap.parse_args()

    given = Path(args.input_file)
    input_file = given if given.exists() else args.res_dir / args.input_file
    if not input_file.exists():
        raise SystemExit(f"input not found: {given} nor {args.res_dir / args.input_file}")
    scores_file = args.out or (args.scores_dir / Path(args.input_file).name)

    concept_map = read_concept_map(args.concepts)
    print(f"judge   {args.model}")
    print(f"input   {input_file}")
    print(f"output  {scores_file}")
    print(f"concept CSV {args.concepts}（{len(concept_map)} 个 concept）"
          + (f"，只判前 {args.limit} 条" if args.limit else ""))

    scores = evaluate(input_file, scores_file, args.model, concept_map,
                      args.cache_dir, args.cache_tag, args.limit)

    print(f"\n{'factor':<10}{'lm_judge':>10}{'concept':>10}"
          f"{'instruction':>13}{'fluency':>10}")
    for i, factor in enumerate(scores["factor"]):
        print(f"{factor:<10}{scores['lm_judge_rating'][i]:>10.3f}"
              f"{scores['relevance_concept_ratings'][i]:>10.3f}"
              f"{scores['relevance_instruction_ratings'][i]:>13.3f}"
              f"{scores['fluency_ratings'][i]:>10.3f}")
    print(f"\n-> {scores_file}")


if __name__ == "__main__":
    main()
