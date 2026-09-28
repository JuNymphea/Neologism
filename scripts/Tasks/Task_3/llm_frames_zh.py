#!/usr/bin/env python
"""Which syntactic environments can each Chinese Task 3 synonym stand in?

Chinese draws no sharp line between adjectives and verbs -- a property word is a
predicate on its own (他很冷静), and UD Chinese tags most of them VERB -- so a
single POS tag per synonym forces a boundary the language does not have. Instead
each word is judged, blind to any concept's category, against three frames:

    N  名词环境  一个/这种 ___ ; ___ 的数量      (names a thing)
    V  动词环境  正在 ___ ; 已经 ___ 了 ; 不要 ___ (an action or process)
    A  形容词环境 很 ___ ; 非常 ___ 的           (a property or state)

A word may fit several (研究 N+V, 冷静 V+A). Also recorded: valid (a word or
phrase, not a sentence, instruction or garbage) and single (one word, not a
phrase such as 很大 or 最高的). English items -- some vectors answer the Chinese
prompt in English -- are judged against a/the ___, to ___, very ___.

Words come from every *t3zh_synonyms_*.json in Task_3/out plus the Task 3 real
words (controls/zh_words3.json). Results are cached per word in
Task_3/out/zh_frames_llm.json, so later runs only pay for new words.

    OPENAI_API_KEY=... python scripts/Tasks/Task_3/llm_frames_zh.py [--limit 100]
"""
import argparse, asyncio, glob, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "controls"))

PROMPT = """下面是一组词语（大多是中文，少数可能是英文）。对每个词，判断它能否自然地放进下面三类句法环境。按词本身判断，忽略词尾的“的”“地”。

N 名词环境：“一个/一种/这种___”，或作为主语、宾语指称事物（如“___的数量”）
V 动词环境：“正在___”“已经___了”“不要___”，表示动作或过程
A 形容词环境：“很___”“非常___的”，表示性质或状态

一个词可以同时适合多个环境。只要这种用法在现代汉语中自然、常见，就标 1，否则标 0。
英文词按对应的英文环境判断：N 为 “a/the ___”，V 为 “to ___”，A 为 “very ___”。
另外标注：
valid：是一个词或短语则为 1；如果是句子、说明文字、提问、乱码或无意义字符串则为 0（此时 N/V/A 都填 0）。
single：是单个词为 1；如果是短语（如“很大”“最高的”“水分多的”“完全浸透”）为 0。

只输出一个 JSON 数组，不要任何解释。数组长度和顺序与输入一致，每项格式：
{"w": "原词", "N": 0或1, "V": 0或1, "A": 0或1, "valid": 0或1, "single": 0或1}

输入：
"""


def collect(out_dir: Path, words_file: Path):
    import analyze_zh_t13 as Z
    words, seen = [], set()

    def add(w):
        w = w.strip()
        if w and w not in seen:
            seen.add(w); words.append(w)

    for w in [x for v in json.loads(words_file.read_text())["words"].values() for x in v]:
        add(w)
    for f in sorted(glob.glob(str(out_dir / "*t3zh_synonyms_*.json"))):
        for samples in json.load(open(f))["results"].values():
            for s in samples:
                items = s["synonyms"] or Z.en_items(s["raw"])
                for w in items:
                    add(w)
    return words


async def run(words, cache_path, model, batch, concurrency):
    import httpx
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY 未设置")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [w for w in words if w not in cache]
    print(f"{len(words)} words, {len(todo)} not yet cached; model {model}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    stats = {"calls": 0, "in": 0, "out": 0, "bad": 0}
    http = httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=180.0,
                             headers={"Authorization": f"Bearer {key}"})

    async def one(chunk, depth=0):
        body = {"model": model, "temperature": 0.0, "max_completion_tokens": 8000,
                "messages": [{"role": "user", "content": PROMPT + json.dumps(chunk, ensure_ascii=False)}]}
        async with sem:
            for attempt in range(5):
                try:
                    r = await http.post("/chat/completions", json=body)
                except (httpx.TimeoutException, httpx.TransportError):
                    await asyncio.sleep(2 ** attempt); continue
                stats["calls"] += 1
                if r.status_code == 200:
                    break
                if r.status_code in (429,) or r.status_code >= 500:
                    await asyncio.sleep(2 ** attempt); continue
                print(f"  API {r.status_code}: {r.text[:160]}"); return
            else:
                return
        data = r.json(); u = data.get("usage") or {}
        stats["in"] += u.get("prompt_tokens", 0); stats["out"] += u.get("completion_tokens", 0)
        text = data["choices"][0]["message"]["content"] or ""
        m = re.search(r"\[.*\]", text, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        got = 0
        if len(items) == len(chunk):
            for w, it in zip(chunk, items):
                if isinstance(it, dict) and all(k in it for k in ("N", "V", "A", "valid", "single")):
                    cache[w] = {k: int(bool(it[k])) for k in ("N", "V", "A", "valid", "single")}
                    got += 1
        if got < len(chunk):
            stats["bad"] += 1
            rest = [w for w in chunk if w not in cache]
            if depth < 2 and len(rest) > 1:      # split and retry what did not come back
                h = len(rest) // 2
                await asyncio.gather(one(rest[:h], depth + 1), one(rest[h:], depth + 1))

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    for i in range(0, len(chunks), 40):
        await asyncio.gather(*(one(c) for c in chunks[i:i + 40]))
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"  {min(i + 40, len(chunks))}/{len(chunks)} batches, cached {len(cache)}; "
              f"calls {stats['calls']} tokens in {stats['in']} out {stats['out']} bad {stats['bad']}", flush=True)
    await http.aclose()
    missing = [w for w in words if w not in cache]
    print(f"done: {len(words) - len(missing)}/{len(words)} judged; tokens in {stats['in']}, out {stats['out']}")
    if missing:
        print("  not judged:", missing[:20])
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out")
    ap.add_argument("--words", type=Path, default=HERE.parent / "controls" / "zh_words3.json")
    ap.add_argument("--cache", type=Path, default=HERE / "out" / "zh_frames_llm.json")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="pilot: only the first N words")
    a = ap.parse_args()
    words = collect(a.out_dir, a.words)
    if a.limit:
        words = words[:a.limit]
    asyncio.run(run(words, a.cache, a.model, a.batch, a.concurrency))


if __name__ == "__main__":
    main()
