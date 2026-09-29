#!/usr/bin/env python
"""Settle the frame ties that more samples could not: by the synonyms' primary use.

After five rounds of five extra samples a few Chinese Task 3 readings still tie,
because every synonym fits the tied frames equally (责备 and 批评 are both noun
and verb; 牢固 and 结实 fit neither noun nor verb). For each such vector, and
only among the tied categories, the LLM is asked which is each synonym's primary,
most common use; the synonyms vote, weighted by how often they were generated,
and the most-voted category is the call. Where no synonym fits either tied
frame (the two-way noun/verb tie of an all-adjective list) the question is a
forced choice, and the decision is marked forced.

Writes Task_3/out/zh_tiebreak_primary.json:
    {file_stem: {label: {"3": cat|null, "2": cat|null, "forced2": bool, "votes": {...}}}}

    OPENAI_API_KEY=... python scripts/Tasks/Task_3/tiebreak_primary_zh.py
"""
import asyncio, collections, glob, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "controls"))
import analyze_zh_frames as ZF  # noqa: E402
import analyze_zh_t13 as Z  # noqa: E402

NAMES = {"N": "名词", "V": "动词", "A": "形容词"}
CAT = {"N": "noun", "V": "verb", "A": "adj"}
PROMPT = """下面每个词可能有多种用法。请只在给定的选项中，判断它在现代汉语里最主要、最常见的用法是哪一种（即使它的其他用法也成立，或者选项都不完全合适，也必须选一个最接近的）。选项：{opts}。
英文词按英语的用法判断。只输出一个 JSON 对象，键为原词，值为 {keys} 中的一个字母，不要解释。
词：{words}"""


def ties(c):
    out = {}
    m3 = max(c)
    top3 = [f for f, x in zip("NVA", c) if x >= m3 - 1e-9]
    if len(top3) > 1:
        out["3"] = top3
    return out


async def ask(http, model, words, opts):
    prompt = PROMPT.format(opts="、".join(f"{o}={NAMES[o]}" for o in opts), keys="/".join(opts),
                           words=json.dumps(words, ensure_ascii=False))
    body = {"model": model, "temperature": 0.0, "max_completion_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(5):
        r = await http.post("/chat/completions", json=body)
        if r.status_code == 200:
            break
        await asyncio.sleep(2 ** attempt)
    text = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", text, re.S)
    got = json.loads(m.group(0)) if m else {}
    return {w: v for w, v in got.items() if v in opts}


async def main():
    import httpx
    cache = json.load(open(HERE / "out/zh_frames_llm.json"))
    http = httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=180.0,
                             headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    out = {}
    for f in sorted(glob.glob(str(HERE / "out/*t3zh_synonyms_*.json"))):
        if "_extra" in f:
            continue
        stem = Path(f).stem
        S, C = ZF.load_samples(f), ZF.compat(f, cache)
        for k, c in C.items():
            if c is None:
                continue
            todo = {}
            t = ties(c)
            if "3" in t:
                todo["3"] = t["3"]
            if k.split("_")[1] in "nv" and abs(c[0] - c[1]) < 1e-9:
                todo["2"] = ["N", "V"]
            if not todo:
                continue
            freq = collections.Counter(w.strip() for s in S[k] for w in (s["synonyms"] or Z.en_items(s["raw"]))
                                       if cache.get(w.strip(), {}).get("valid"))
            rec = {"3": None, "2": None, "forced2": False, "votes": {}}
            for way, opts in todo.items():
                fit = {w: n for w, n in freq.items() if any(cache[w][o] for o in opts)}
                forced = not fit
                words = list(fit or freq)
                prim = await ask(http, "gpt-5.4-nano", words, opts)
                votes = collections.Counter()
                for w, o in prim.items():
                    votes[o] += freq[w]
                best = max(opts, key=lambda o: (votes[o], -opts.index(o)))
                rec[way] = CAT[best]
                rec["votes"][way] = dict(votes)
                if way == "2":
                    rec["forced2"] = forced
            out.setdefault(stem, {})[k] = rec
            print(f"{stem:32} {k:18} " + "  ".join(f"{w}-way -> {rec[w]} {rec['votes'].get(w)}" for w in todo)
                  + ("  [forced]" if rec["forced2"] else ""))
    await http.aclose()
    (HERE / "out/zh_tiebreak_primary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", HERE / "out/zh_tiebreak_primary.json")


if __name__ == "__main__":
    asyncio.run(main())
