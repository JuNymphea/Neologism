"""Reading Universal Dependencies CoNLL-U treebanks.

A deliberately small, dependency-free reader: the induction pipeline should be
auditable end-to-end, and CoNLL-U is simple enough that a third-party parser
buys us nothing but a version pin.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List

# Sentence-boundary pseudo-tokens used when a context window runs off the edge.
BOS = "<s>"
EOS = "</s>"


@dataclass
class Token:
    idx: int  # 0-based index within the sentence
    form: str
    lemma: str
    upos: str
    feats: str
    head: int  # 1-based UD head id; 0 means root
    deprel: str
    space_after: bool


@dataclass
class Sentence:
    sent_id: str
    source: str  # e.g. "ewt-train"
    tokens: List[Token] = field(default_factory=list)
    text: str = ""

    def __len__(self) -> int:
        return len(self.tokens)


def _parse_block(lines: List[str], source: str) -> Sentence | None:
    sent = Sentence(sent_id="", source=source)
    for line in lines:
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sent.sent_id = line.split("=", 1)[-1].strip()
            elif line.startswith("# text"):
                sent.text = line.split("=", 1)[-1].strip()
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            continue
        tok_id = cols[0]
        # Skip multiword-token ranges ("1-2") and empty nodes ("3.1"); the
        # basic layer we want is carried by the plain integer ids.
        if "-" in tok_id or "." in tok_id:
            continue
        try:
            head = int(cols[6]) if cols[6] != "_" else 0
        except ValueError:
            head = 0
        sent.tokens.append(
            Token(
                idx=len(sent.tokens),
                form=cols[1],
                lemma=cols[2] if cols[2] != "_" else cols[1].lower(),
                upos=cols[3],
                feats=cols[5],
                head=head,
                deprel=cols[7],
                space_after="SpaceAfter=No" not in cols[9],
            )
        )
    return sent if sent.tokens else None


def read_conllu(path: str | Path, source: str | None = None) -> Iterator[Sentence]:
    """Yield every sentence in a .conllu file."""
    path = Path(path)
    if source is None:
        source = path.stem
    block: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if block:
                    sent = _parse_block(block, source)
                    if sent is not None:
                        yield sent
                    block = []
            else:
                block.append(line)
    if block:
        sent = _parse_block(block, source)
        if sent is not None:
            yield sent


def load_corpus(paths: List[str | Path]) -> List[Sentence]:
    """Load several treebank files into one in-memory corpus."""
    sentences: List[Sentence] = []
    for p in paths:
        sentences.extend(read_conllu(p))
    return sentences









def form_distributions(sentences: List[Sentence]) -> dict:
    """form -> set of (UPOS, deprel) it is attested in.

    Used to check that a generic filler actually occurs in the syntactic
    position a frame puts it in, rather than merely having the right UPOS.
    """
    out: dict = {}
    for sent in sentences:
        for tok in sent.tokens:
            out.setdefault(tok.form.lower(), set()).add((tok.upos, tok.deprel))
    return out


def bigram_counts(sentences: List[Sentence]) -> Counter:
    """Adjacent word-form bigrams over the pooled treebanks.

    A corpus-internal naturalness check for the material we add around the
    frame: "they tried to" is attested thousands of times, "the same to" is
    not. No language model is involved -- using one would be circular, since
    Task 2 measures LM surprisal.
    """
    counts: Counter = Counter()
    for sent in sentences:
        forms = [t.form.lower() for t in sent.tokens]
        for a, b in zip(forms, forms[1:]):
            counts[(a, b)] += 1
    return counts




def frequent_adverbs(sentences: List[Sentence], top_k: int) -> set[str]:
    """The `top_k` most frequent ADV word forms.

    Degree and polarity adverbs ("very", "too", "not", "really") are among the
    strongest grammatical cues for adjective slots, but UD tags them ADV
    alongside contentful adverbs like "carefully". Rather than hand-listing the
    grammatical ones -- exactly the researcher intuition this pipeline exists to
    remove -- we lexicalize whichever ADV forms the corpus itself makes frequent
    and abstract the rest.
    """
    counts: Counter = Counter()
    for sent in sentences:
        for tok in sent.tokens:
            if tok.upos == "ADV":
                counts[tok.form.lower()] += 1
    return {form for form, _ in counts.most_common(top_k)}
