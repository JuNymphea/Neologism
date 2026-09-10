#!/usr/bin/env python
"""Build the control-word set for the Task 2 probe.

Task 2 needs known words at three separate stages, and each has to be a
different set of words:

    calibration (150/POS)  fits the matched and non-matched reference
                           distributions for each candidate
    probe-dev    (75/POS)  decides which candidates are diagnostic enough to
                           enter the final slot set
    final-test   (75/POS)  reports the frozen probe's accuracy -- never
                           touched during fitting or slot selection

Two sets are not enough. With only calibration and validation, the slot set
ends up chosen on the same words the accuracy is reported on, so that accuracy
is optimistic. The original setup collapsed all three into one.

Four independent things have to be right, and any one of them getting through
wrong quietly invalidates the calibration:

1. **Single token, in the position the probe uses.** Every probe item places
   the word after a space, so `" word"` -- not `"word"` -- must encode to one
   id. The earlier lists tested vocabulary membership instead, which let
   through 75/300 nouns that are two tokens after a space (`" abbot"` ->
   `▁ab` + `bot`, because only `▁Abbot` is in the vocabulary).

2. **Unambiguous part of speech**, so `plate`, `run` and `dark` are out.

3. **The inflection the slot requires.** All thirty slots want a bare form, and
   the script verifies that rather than assuming it: in English the base verb
   form serves as infinitive *and* 1sg present, while bare noun and adjective
   forms are singular and positive-degree. Five slots additionally read
   `A {SLOT} ...`, which a mass noun would make ungrammatical.

4. **Comparable frequency across the three categories.** A frequent word has
   lower continuation surprisal whatever its category, so if the noun words
   were systematically rarer than the verb words the probe would measure
   frequency and report it as a category effect.

    python Task_2/build_control_words.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

POS_KEYS = ("noun", "verb", "adj")

#: Which UD feature each slot demands, and why a bare form satisfies it.
MORPH_SATISFIED_BY_BARE = {
    "Number=Sing": "bare noun form is singular",
    "Degree=Pos": "bare adjective form is positive degree",
    "VerbForm=Inf": "bare verb form is the infinitive",
    "Mood=Ind": "English base form doubles as 1sg present indicative",
}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def token_id(tok, word: str) -> int | None:
    """The single token id for `" word"`, or None if it is not one token."""
    ids = tok.encode(" " + word, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None


def noun_countability(ud_dir: Path) -> Tuple[set, set]:
    """Nouns attested with an indefinite article in the treebanks.

    Five probe items read `A {SLOT} ...`, where a mass noun ("A water to")
    is ungrammatical and would manufacture surprisal unrelated to category.
    WordNet does not record countability, so this asks the corpus directly.

    Returns (seen_as_noun, seen_with_indefinite_article). A word in the first
    set but not the second was used often enough to judge and never took
    "a"/"an", which is the mass-noun signature. A word in neither cannot be
    judged, so it is kept.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slotgen.ud_io import load_corpus

    files = sorted(ud_dir.glob("*.conllu"))
    if not files:
        return set(), set()
    seen: Counter = Counter()
    with_article: set = set()
    for sent in load_corpus(files):
        for tok in sent.tokens:
            if tok.upos == "NOUN":
                seen[tok.lemma.lower()] += 1
            if tok.deprel == "det" and tok.form.lower() in {"a", "an"}:
                head = tok.head - 1
                if 0 <= head < len(sent.tokens) and sent.tokens[head].upos == "NOUN":
                    with_article.add(sent.tokens[head].lemma.lower())
    # Judge only nouns the corpus shows often enough for the absence of an
    # article to mean something.
    judgeable = {w for w, n in seen.items() if n >= 5}
    return judgeable, with_article


def check_slot_morphology(slots_path: Path) -> dict:
    """Confirm every slot's requirement is met by a bare form."""
    data = json.loads(slots_path.read_text())
    reqs: Counter = Counter()
    for pos in data:
        for item in data[pos]:
            reqs[item.get("morph_requirement", "Any").split("|")[0]] += 1
    unmet = [r for r in reqs if r not in MORPH_SATISFIED_BY_BARE and r != "Any"]
    return {"counts": dict(reqs), "unmet": unmet,
            "rationale": MORPH_SATISFIED_BY_BARE}


# ---------------------------------------------------------------------------
# Frequency matching
# ---------------------------------------------------------------------------


def assign_bins(freqs: Dict[str, int], n_bins: int) -> Dict[str, int]:
    """Bin words by frequency rank; lower token id means a commoner piece."""
    ordered = sorted(freqs, key=lambda w: freqs[w])
    n = len(ordered)
    return {w: min(n_bins - 1, i * n_bins // n) for i, w in enumerate(ordered)}


def bin_quota(by_bin: Dict[str, Dict[int, List[str]]], n_bins: int,
              target: int) -> Tuple[Dict[int, int], int]:
    """How many words to draw from each stratum.

    A stratum can supply only as many words as its scarcest category holds, so
    the quota is capped by that and shared out in proportion. This is what
    keeps the three categories' frequency profiles aligned; the binding
    constraint is verbs, of which the vocabulary holds far fewer.
    """
    capacity = {b: min(len(by_bin[p][b]) for p in POS_KEYS) for b in range(n_bins)}
    total = sum(capacity.values())
    quota = {b: min(capacity[b], round(target * capacity[b] / total))
             for b in range(n_bins)} if total else {}
    while sum(quota.values()) < target:
        room = [b for b in range(n_bins) if quota[b] < capacity[b]]
        if not room:
            break
        quota[max(room, key=lambda b: capacity[b] - quota[b])] += 1
    while sum(quota.values()) > target:
        used = [b for b in range(n_bins) if quota[b] > 0]
        quota[max(used, key=lambda b: quota[b])] -= 1
    return quota, total


def frequency_match_report(selected: Dict[str, List[str]],
                           freqs: Dict[str, int]) -> dict:
    """Evidence that the three categories ended up equally frequent."""
    out: dict = {"quantiles": {}, "ks_tests": {}}
    for pos in POS_KEYS:
        ids = sorted(freqs[w] for w in selected[pos])
        out["quantiles"][pos] = {
            "n": len(ids),
            "q1": ids[len(ids) // 4],
            "median": int(st.median(ids)),
            "q3": ids[3 * len(ids) // 4],
        }
    try:
        from scipy.stats import ks_2samp

        for a in range(3):
            for b in range(a + 1, 3):
                p, q = POS_KEYS[a], POS_KEYS[b]
                r = ks_2samp([freqs[w] for w in selected[p]],
                             [freqs[w] for w in selected[q]])
                out["ks_tests"][f"{p}_vs_{q}"] = {
                    "statistic": round(float(r.statistic), 4),
                    "p_value": round(float(r.pvalue), 4),
                }
    except ImportError:
        out["ks_tests"] = "scipy unavailable"
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-words", type=Path, default=root / "pos_words_v2.json")
    ap.add_argument("--tokenizer", type=Path, default=root / "tokenizer.json")
    ap.add_argument("--slots", type=Path, default=here / "out" / "slots_flat.json")
    ap.add_argument("--ud-dir", type=Path, default=here / "data" / "ud")
    ap.add_argument("--out-calibration", type=Path,
                    default=here / "out" / "control_calibration.json")
    ap.add_argument("--out-probedev", type=Path,
                    default=here / "out" / "control_probedev.json")
    ap.add_argument("--out-finaltest", type=Path,
                    default=here / "out" / "control_finaltest.json")
    ap.add_argument("--out-meta", type=Path,
                    default=here / "out" / "control_words_meta.json",
                    help="filter counts, frequency-match evidence, provenance")
    ap.add_argument("--n-calibration", type=int, default=150,
                    help="words per POS used to fit the reference distributions")
    ap.add_argument("--n-probedev", type=int, default=75,
                    help="words per POS used only to screen candidate slots")
    ap.add_argument("--n-finaltest", type=int, default=75,
                    help="words per POS used only for the final frozen report")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="frequency strata used to match the three categories")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--skip-countability", action="store_true")
    args = ap.parse_args()

    target = args.n_calibration + args.n_probedev + args.n_finaltest
    rng = random.Random(args.seed)
    tok = load_tokenizer(args.tokenizer)
    words = json.loads(args.pos_words.read_text())
    filters: Dict[str, dict] = {p: {"input": len(words[p])} for p in POS_KEYS}

    # -- 0. do the slots all want the same inflection? -----------------------
    morph = check_slot_morphology(args.slots)
    print("Slot morphology:", morph["counts"])
    if morph["unmet"]:
        print(f"  !! not satisfied by a bare form: {morph['unmet']}")
    else:
        print("  all satisfied by bare forms -> one control set covers all slots")

    # -- 1. single token where the probe puts it -----------------------------
    pool: Dict[str, Dict[str, int]] = {}
    for pos in POS_KEYS:
        pool[pos] = {w: t for w in words[pos] if (t := token_id(tok, w)) is not None}
        filters[pos]["single_token"] = len(pool[pos])

    # -- 2. countability, for the five slots with an indefinite article ------
    if not args.skip_countability:
        judgeable, with_article = noun_countability(args.ud_dir)
        if judgeable:
            before = len(pool["noun"])
            dropped = sorted(w for w in pool["noun"]
                             if w in judgeable and w not in with_article)
            pool["noun"] = {w: i for w, i in pool["noun"].items()
                            if w not in set(dropped)}
            filters["noun"]["mass_nouns_dropped"] = len(dropped)
            filters["noun"]["after_countability"] = len(pool["noun"])
            print(f"  countability: {before} -> {len(pool['noun'])} "
                  f"(dropped {len(dropped)} likely mass nouns, "
                  f"e.g. {', '.join(dropped[:6])})")

    # -- 3. frequency-matched stratified sample ------------------------------
    freqs = {w: i for pos in POS_KEYS for w, i in pool[pos].items()}
    bins = assign_bins(freqs, args.n_bins)
    by_bin = {p: {b: [] for b in range(args.n_bins)} for p in POS_KEYS}
    for pos in POS_KEYS:
        for w in pool[pos]:
            by_bin[pos][bins[w]].append(w)
    quota, capacity_total = bin_quota(by_bin, args.n_bins, target)
    if capacity_total < target:
        print(f"\n!! matched capacity is {capacity_total}/POS, below the "
              f"requested {target}")

    print(f"\nfrequency strata ({args.n_bins} bins over token-id rank)")
    print(f"  {'bin':<5}{'noun':>7}{'verb':>7}{'adj':>7}{'quota':>8}")
    for b in range(args.n_bins):
        print(f"  {b:<5}" + "".join(f"{len(by_bin[p][b]):>7}" for p in POS_KEYS)
              + f"{quota[b]:>8}")

    selected = {p: sorted(w for b in range(args.n_bins)
                          for w in rng.sample(by_bin[p][b], quota[b]))
                for p in POS_KEYS}

    # -- 4. three-way split, stratified so all three match on frequency ------
    splits = {"calibration": {p: [] for p in POS_KEYS},
              "probedev": {p: [] for p in POS_KEYS},
              "finaltest": {p: [] for p in POS_KEYS}}
    for pos in POS_KEYS:
        for b in range(args.n_bins):
            in_bin = [w for w in selected[pos] if bins[w] == b]
            rng.shuffle(in_bin)
            n = len(in_bin)
            a = round(n * args.n_calibration / target)
            c = round(n * args.n_probedev / target)
            splits["calibration"][pos] += in_bin[:a]
            splits["probedev"][pos] += in_bin[a:a + c]
            splits["finaltest"][pos] += in_bin[a + c:]
        for s in splits:
            splits[s][pos].sort()
        names = list(splits)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not set(splits[names[i]][pos]) & set(splits[names[j]][pos]), pos
    calibration, probedev, finaltest = (splits["calibration"], splits["probedev"],
                                        splits["finaltest"])

    # -- 5. report -----------------------------------------------------------
    match = frequency_match_report(selected, freqs)
    print(f"\n{'':6}{'calib':>7}{'dev':>6}{'test':>6}{'total':>7}"
          f"{'median id':>12}{'IQR':>20}")
    for pos in POS_KEYS:
        q = match["quantiles"][pos]
        iqr = "%d-%d" % (q["q1"], q["q3"])
        print(f"{pos:<6}{len(calibration[pos]):>7}{len(probedev[pos]):>6}"
              f"{len(finaltest[pos]):>6}{q['n']:>7}{q['median']:>12}{iqr:>20}")
    if isinstance(match["ks_tests"], dict):
        print("\n频率分布一致性 (Kolmogorov-Smirnov, p>0.05 表示无显著差异):")
        for k, v in match["ks_tests"].items():
            flag = "ok" if v["p_value"] > 0.05 else "!! 分布有差异"
            print(f"  {k:<16} D={v['statistic']:.3f}  p={v['p_value']:.3f}  {flag}")

    # Two word files, so that neither can be reached for by accident: the
    # calibration set fits the distributions, the validation set is only ever
    # read when reporting accuracy and deciding which slots to drop.
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    provenance = {
        "config": config,
        "frequency_source": "gemma token id (SentencePiece unigram rank)",
        "seed": args.seed,
    }

    def dump(path: Path, role: str, words: Dict[str, List[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "role": role,
            "n_per_pos": {p: len(words[p]) for p in POS_KEYS},
            **provenance,
            "words": words,
            "token_ids": {w: freqs[w] for p in POS_KEYS for w in words[p]},
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    dump(args.out_calibration,
         "calibration: fit reference distributions only", calibration)
    dump(args.out_probedev,
         "probe-dev: screen candidate slots for diagnosticity; never fitted on, "
         "never used for the final report", probedev)
    dump(args.out_finaltest,
         "final-test: the frozen probe's accuracy; touched at no earlier stage",
         finaltest)

    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps({
        **provenance,
        "slot_morphology": morph,
        "filters": filters,
        "frequency_match": match,
        "bin_quota": quota,
        "n_per_split": {s: {p: len(splits[s][p]) for p in POS_KEYS} for s in splits},
        "disjoint": True,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n-> {args.out_calibration}   (拟合参考分布)")
    print(f"-> {args.out_probedev}   (筛选候选 slot)")
    print(f"-> {args.out_finaltest}   (冻结后最终报告)")
    print(f"-> {args.out_meta}   (筛选统计与频率匹配证据)")


if __name__ == "__main__":
    main()
