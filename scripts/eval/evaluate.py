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
import re
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

from language_models import LanguageModel
from lm_judge import LMJudgeEvaluator

DEFAULT_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_CACHE_TAG = "demo"
CONCEPT_CSV_NAME = "pos - FrameNet.csv"

EVAL_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EVAL_DIR.parent

#: The two kinds of generation file this judges, and where each lives. Both hold
#: rows of {question, normal_answer, concept_answer}: `train` is the data the
#: vectors were trained towards, `results` is what the trained vectors produced.
SOURCES = {
    "train": SCRIPTS_DIR / "train" / "data" / "train",
    "results": SCRIPTS_DIR / "train" / "results",
}
#: Scores are written under here, one subfolder per source, so the two never mix.
DEFAULT_SCORES_DIR = EVAL_DIR / "scores"
#: Judge replies cached between runs, keyed by prompt text. Point JUDGE_CACHE_DIR
#: (or --cache-dir) at an existing cached_data/ to reuse judgements already paid for.
DEFAULT_CACHE_DIR = os.environ.get("JUDGE_CACHE_DIR") or str(EVAL_DIR / "cached_data")

TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")
#: A concept id is a POS letter and a number standing as its own underscore field:
#: "a_1", "n_16_gemma_a25" -> n_16, "gemma_a_1_unbiased" -> a_1.
CONCEPT_RE = re.compile(r"(?:^|_)([anv]_\d+)(?=_|$)")


def natural_key(path: Path):
    """a_2 before a_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def default_concept_csv() -> Path:
    """The copy next to this script, unless the original under scripts/framenet/ exists."""
    candidates = [SCRIPTS_DIR / "framenet" / CONCEPT_CSV_NAME, EVAL_DIR / CONCEPT_CSV_NAME]
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


def parse_file_name(input_file: Path, concept_map: dict) -> tuple[str, str | None]:
    """(concept id, template or None) from a generation file's name.

    The id used to be "the first two underscore fields", which is right for
    `a_1.jsonl` and `n_16_gemma_a25.jsonl` but turns the sweep's
    `gemma_a_1_unbiased.jsonl` into `gemma_a`. It is now the first field pair
    that looks like an id *and* is in the concept CSV, so a model prefix is
    skipped over rather than mistaken for part of the id.
    """
    stem = input_file.stem
    for m in CONCEPT_RE.finditer(stem):
        concept = m.group(1)
        if concept in concept_map:
            tail = stem[m.end():].strip("_").split("_")
            template = tail[-1] if tail and tail[-1] in TEMPLATES else None
            return concept, template
    raise KeyError(
        f"no concept id from the CSV in '{input_file.name}'; expected a name like "
        f"'a_1.jsonl' or 'gemma_a_1_unbiased.jsonl'. Known ids include: "
        f"{', '.join(sorted(concept_map)[:8])}...")


def transform_res_data(input_file: Path, concept: str, concept_map: dict,
                       limit: int | None = None) -> pd.DataFrame:
    """One judged row per answer: each question contributes normal + concept."""
    df = pd.read_json(input_file, lines=True)
    if limit:
        df = df.head(limit)

    missing = {"question", "normal_answer", "concept_answer"} - set(df.columns)
    if missing:
        raise KeyError(f"{input_file.name} lacks {sorted(missing)}")

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


def close_client(lm_model: LanguageModel) -> None:
    try:
        asyncio.run(lm_model.close())
    except RuntimeError:
        # Expected: chat_completions() ran asyncio.run per batch, so the loop
        # the client bound to is long gone. The scores are already on disk.
        pass


def judge_file(lm_model: LanguageModel, model: str, input_file: Path, concept: str,
               concept_map: dict, limit: int | None = None) -> dict:
    """The judge's metrics for one file -- unchanged from the original evaluator."""
    evaluator = LMJudgeEvaluator(lm_model=lm_model, model_name=model)
    return evaluator.compute_metrics(
        transform_res_data(input_file, concept, concept_map, limit))


def evaluate(input_file: Path, scores_file: Path, model: str,
             concept_map: dict, cache_dir: str = DEFAULT_CACHE_DIR,
             cache_tag: str = DEFAULT_CACHE_TAG,
             limit: int | None = None) -> dict:
    """Judge a single file end to end. Kept for callers that import this module."""
    lm_model = LanguageModel(model=model, client=get_chat_client(model), use_cache=True,
                             cache_level="prompt", master_data_dir=cache_dir,
                             cache_tag=cache_tag)
    concept, _ = parse_file_name(Path(input_file), concept_map)
    scores = judge_file(lm_model, model, Path(input_file), concept, concept_map, limit)
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    scores_file.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    lm_model.save_cache()
    close_client(lm_model)
    return scores


def infer_source(path: Path) -> str:
    resolved = path.resolve()
    for name, root in SOURCES.items():
        if resolved.parent == root.resolve():
            return name
    return "custom"


def resolve_inputs(items: list[str], source: str | None, take_all: bool) -> list[Path]:
    """Paths, bare names, or glob patterns; names and patterns are looked up in the source dir."""
    found: list[Path] = []
    source_dir = SOURCES.get(source) if source else None
    if take_all:
        if source_dir is None:
            raise SystemExit("--all needs --source train or --source results")
        found += sorted(source_dir.glob("*.jsonl"), key=natural_key)
    for item in items or []:
        given = Path(item)
        if given.exists():
            found.append(given)
            continue
        if source_dir is None:
            raise SystemExit(f"input not found: {item}  (pass --source to look it up by name)")
        pattern = item if item.endswith(".jsonl") else f"{item}.jsonl"
        matches = sorted(source_dir.glob(pattern), key=natural_key)
        if not matches:
            raise SystemExit(f"nothing in {source_dir} matches {pattern}")
        found += matches
    unique = list(dict.fromkeys(p.resolve() for p in found))
    if not unique:
        raise SystemExit("no input files; pass --input-file or --all")
    return unique


def factor_row(scores: dict, factor: str) -> dict:
    """Pick a factor's means by name -- groupby sorts them, so position is not stable."""
    if factor not in scores.get("factor", []):
        return {}
    i = scores["factor"].index(factor)
    return {
        "lm_judge": scores["lm_judge_rating"][i],
        "concept": scores["relevance_concept_ratings"][i],
        "instruction": scores["relevance_instruction_ratings"][i],
        "fluency": scores["fluency_ratings"][i],
    }


def write_summary(source_dir: Path) -> Path | None:
    """Rebuild summary.csv from every score file in one source folder.

    Rebuilt from disk rather than appended to, so it always matches the score
    files exactly -- re-judging or deleting a file is reflected on the next run.
    """
    files = sorted(source_dir.glob("*.json"), key=natural_key)
    rows = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        meta = data.get("meta")
        if not meta:
            continue
        normal, concept = factor_row(data, "normal"), factor_row(data, "concept")
        row = {"file": meta["input_file"], "concept": meta["concept"],
               "template": meta.get("template") or "", "judge": meta["judge"],
               "n_questions": meta["n_questions"]}
        for metric in ("lm_judge", "concept", "instruction", "fluency"):
            row[f"normal_{metric}"] = round(normal.get(metric, float("nan")), 4)
            row[f"concept_{metric}"] = round(concept.get(metric, float("nan")), 4)
        row["gap_lm_judge"] = round(row["concept_lm_judge"] - row["normal_lm_judge"], 4)
        row["gap_concept"] = round(row["concept_concept"] - row["normal_concept"], 4)
        rows.append(row)
    if not rows:
        return None
    out = source_dir / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", "--input_file", dest="input_file", nargs="*", default=[],
                    help="paths, bare names or glob patterns ('a_*', 'gemma_*_mixed'); "
                         "names and patterns are looked up in the --source folder")
    ap.add_argument("--source", choices=sorted(SOURCES),
                    help="train: scripts/train/data/train, results: scripts/train/results")
    ap.add_argument("--all", action="store_true", help="every .jsonl in the --source folder")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concepts", type=Path, default=default_concept_csv())
    ap.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR,
                    help="scores go to <scores-dir>/<source>/<input name>.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit output path, for a single input file only")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--cache-tag", default=DEFAULT_CACHE_TAG)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N questions (2N answers)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-judge files whose scores already exist with the same judge and limit")
    args = ap.parse_args()

    inputs = resolve_inputs(args.input_file, args.source, args.all)
    if args.out and len(inputs) > 1:
        raise SystemExit("--out names one file; drop it when judging several")
    concept_map = read_concept_map(args.concepts)

    plan = []
    for path in inputs:
        source = args.source if args.source and infer_source(path) in (args.source, "custom") \
            else infer_source(path)
        concept, template = parse_file_name(path, concept_map)
        out = args.out or (args.scores_dir / source / f"{path.stem}.json")
        plan.append((path, source, concept, template, out))

    print(f"judge   {args.model}")
    print(f"cache   {Path(args.cache_dir) / 'persist_lm_cache'}")
    print(f"concept CSV {args.concepts}（{len(concept_map)} 个 concept）"
          + (f"，每个文件只判前 {args.limit} 条" if args.limit else ""))
    print(f"{len(plan)} 个文件")

    lm_model = None
    touched = set()
    try:
        for k, (path, source, concept, template, out) in enumerate(plan, 1):
            n_questions = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            n_questions = min(n_questions, args.limit) if args.limit else n_questions
            tag = f"[{k}/{len(plan)}] {source}/{path.name}"

            if out.exists() and not args.overwrite:
                old = json.loads(out.read_text(encoding="utf-8")).get("meta", {})
                if old.get("judge") == args.model and old.get("n_questions") == n_questions:
                    print(f"{tag} -- 已有同 judge、同题数的结果，跳过")
                    touched.add(out.parent)
                    continue

            print(f"\n{tag}  concept={concept}"
                  + (f"  template={template}" if template else "")
                  + f"  {n_questions} 题 -> 最多 {n_questions * 2 * 3} 次 judge 调用")
            if lm_model is None:
                # Built once: the cache pickle is hundreds of MB and would otherwise be
                # reloaded per file. The HTTP client is made fresh per file instead,
                # because the judge closes its event loop after every batch.
                lm_model = LanguageModel(model=args.model, client=get_chat_client(args.model),
                                         use_cache=True, cache_level="prompt",
                                         master_data_dir=args.cache_dir,
                                         cache_tag=args.cache_tag)
            else:
                lm_model.client = get_chat_client(args.model)

            scores = judge_file(lm_model, args.model, path, concept, concept_map, args.limit)
            scores["meta"] = {
                "source": source,
                "input_file": path.name,
                "input_path": str(path),
                "concept": concept,
                "concept_description": concept_map[concept],
                "template": template,
                "judge": args.model,
                "n_questions": n_questions,
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
            touched.add(out.parent)
            # Saved after every file so an interrupted batch keeps what it paid for.
            lm_model.save_cache()
            close_client(lm_model)

            print(f"{'factor':<10}{'lm_judge':>10}{'concept':>10}{'instruction':>13}{'fluency':>10}")
            for factor in ("normal", "concept"):
                r = factor_row(scores, factor)
                if r:
                    print(f"{factor:<10}{r['lm_judge']:>10.3f}{r['concept']:>10.3f}"
                          f"{r['instruction']:>13.3f}{r['fluency']:>10.3f}")
            print(f"-> {out}")
    finally:
        if lm_model is not None:
            lm_model.save_cache()

    for folder in sorted(touched):
        summary = write_summary(folder)
        if summary:
            print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()
