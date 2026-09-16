#!/usr/bin/env python
"""Score concept relevance alone, skipping instruction relevance and fluency.

The full judge asks three questions per answer and combines them by harmonic
mean. When the question is only "did the concept get in", two thirds of that is
wasted: 50 items cost 300 calls instead of 50.

The rubric is `UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE`,
used verbatim, so a number here means the same thing as the
`relevance_concept_ratings` column of a full run *by the same judge*. It is not
comparable across judges -- the English set was scored by gpt-5.2-2025-12-11,
so scores from another model belong in their own column.

    export OPENAI_API_KEY=...
    python score_concept_only.py --input-file out/*.jsonl \
        --factors concept --limit 50 --model gpt-5.4-nano

`--factors normal,concept` also scores the plain answers, which is what gives
the floor: how much of the concept turns up when nobody asked for it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics as st
from pathlib import Path

from evaluate import default_concept_csv, parse_file_name, read_concept_map
from prompt_templates import UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE as TPL

HERE = Path(__file__).resolve().parent
DEFAULT_RATING = 0.0


def parse_rating(completion: str) -> float:
    """Same extraction as LMJudgeEvaluator, so the numbers line up."""
    if "Rating:" not in completion:
        return DEFAULT_RATING
    t = completion.split("Rating:")[-1].strip().split("\n")[0]
    t = t.replace("[", "").replace("]", "").rstrip(".").strip('"').strip("'").strip("*").strip()
    try:
        r = float(t)
    except ValueError:
        return DEFAULT_RATING
    return r if 0.0 <= r <= 2.0 else DEFAULT_RATING


class Judge:
    def __init__(self, model, cache_path: Path, concurrency=16):
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY 未设置")
        self.model, self.cache_path = model, cache_path
        self.calls = self.tok_in = self.tok_out = 0
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._dirty = 0
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(
            base_url="https://api.openai.com/v1", timeout=120.0,
            headers={"Authorization": f"Bearer {key}"},
            limits=httpx.Limits(max_keepalive_connections=concurrency,
                                max_connections=concurrency * 2))

    async def ask(self, prompt: str) -> str:
        import httpx
        ck = f"{self.model}|{hashlib.sha1(prompt.encode()).hexdigest()}"
        if ck in self.cache:
            return self.cache[ck]
        body = {"model": self.model, "temperature": 0.0,
                "max_completion_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}]}
        async with self.sem:
            for attempt in range(5):
                try:
                    r = await self.http.post("/chat/completions", json=body)
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt == 4:
                        return ""
                    await asyncio.sleep(2 ** attempt)
                    continue
                self.calls += 1
                if r.status_code == 200:
                    break
                if (r.status_code >= 500 or r.status_code == 429) and attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"  API {r.status_code}: {r.text[:120]}")
                return ""
        data = r.json()
        u = data.get("usage") or {}
        self.tok_in += u.get("prompt_tokens", 0)
        self.tok_out += u.get("completion_tokens", 0)
        out = data["choices"][0]["message"]["content"].strip()
        self.cache[ck] = out
        self._dirty += 1
        if self._dirty >= 25:
            self.save()
        return out

    def save(self):
        self._dirty = 0
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


async def score_file(judge, path, concept, desc, rows, factors):
    jobs = [(f, r[f"{f}_answer"]) for r in rows for f in factors if r.get(f"{f}_answer")]

    async def one(factor, sentence):
        return factor, parse_rating(await judge.ask(TPL.format(concept=desc, sentence=sentence)))

    out = await asyncio.gather(*(one(f, s) for f, s in jobs))
    by = {f: [] for f in factors}
    for f, v in out:
        by[f].append(v)
    judge.save()
    print(f"  {path.name:<12}{concept:<6}{len(jobs):>4} 条   "
          + "   ".join(f"{f} {st.mean(by[f]):.2f}" for f in factors if by[f]), flush=True)
    return by


async def main_async(args):
    cmap = read_concept_map(args.concepts)
    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    plan = []
    for path in args.input_file:
        concept, _ = parse_file_name(path, cmap)
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        if args.limit:
            rows = rows[:args.limit]
        plan.append((path, concept, cmap[concept], rows))
    total = sum(len(r) * len(factors) for _, _, _, r in plan)
    print(f"判官 {args.model}｜{len(plan)} 个文件 × {args.limit or '全部'} 条 × "
          f"{len(factors)} 个回答 = {total} 次调用（只评 concept relevance）\n")
    if args.dry_run:
        for path, concept, desc, rows in plan:
            print(f"  {path.name:<12}{concept:<6}{desc[:26]:<28}{len(rows)*len(factors):>5} 次")
        print("\n--dry-run：未调用 API。")
        return

    judge = Judge(args.model, args.cache, args.concurrency)
    summary = []
    for path, concept, desc, rows in plan:
        by = await score_file(judge, path, concept, desc, rows, factors)
        summary.append((path, concept, desc, by))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{path.stem}.json").write_text(json.dumps({
            "input": str(path), "concept": concept, "concept_desc": desc,
            "judge": args.model, "rubric": "concept_relevance_only",
            "n": len(rows), "factors": factors,
            "mean": {f: st.mean(by[f]) for f in by if by[f]},
            "ratings": by}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== concept relevance（0–2）===")
    hdr = f"{'':4}{'文件':<10}{'概念':<26}"
    for f in factors:
        hdr += f"{f:>9}"
    print(hdr + f"{'满分':>8}{'零分':>8}")
    for path, concept, desc, by in summary:
        line = f"  {path.stem:<12}{desc[:24]:<26}"
        for f in factors:
            line += f"{st.mean(by[f]):>9.2f}" if by[f] else f"{'-':>9}"
        c = by.get("concept") or []
        if c:
            line += (f"{sum(1 for x in c if x == 2)/len(c):>8.0%}"
                     f"{sum(1 for x in c if x == 0)/len(c):>8.0%}")
        print(line)
    allc = [x for _, _, _, by in summary for x in (by.get("concept") or [])]
    if allc:
        print(f"\n  合计 {len(allc)} 条：均值 {st.mean(allc):.3f}"
              f"   满分 {sum(1 for x in allc if x == 2)/len(allc):.0%}"
              f"   零分 {sum(1 for x in allc if x == 0)/len(allc):.0%}")
        for pos, lab in (("n", "名词"), ("v", "动词"), ("a", "形容词")):
            v = [x for _, c, _, by in summary if c.startswith(pos)
                 for x in (by.get("concept") or [])]
            if v:
                print(f"    {lab}  {len(v):>4} 条  均值 {st.mean(v):.3f}"
                      f"   零分 {sum(1 for x in v if x == 0)/len(v):.0%}")
    print(f"\n调用 {judge.calls} 次   输入 {judge.tok_in:,} / 输出 {judge.tok_out:,} tokens"
          f"   花费 ${judge.tok_in/1e6*0.20 + judge.tok_out/1e6*1.25:.3f}")
    print(f"-> {args.out_dir}")
    judge.save()
    await judge.http.aclose()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", type=Path, nargs="+", required=True)
    ap.add_argument("--factors", default="concept",
                    help="评哪些回答：concept / normal / normal,concept")
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--concepts", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "scores" / "concept_only")
    ap.add_argument("--cache", type=Path, default=HERE / "cached_data" / "concept_only_cache.json")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.concepts is None:
        a.concepts = default_concept_csv()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
