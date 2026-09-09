"""Steps 1-2: turn every NOUN/VERB/ADJ occurrence into context signatures.

A *context signature* is a short window around a target token in which
grammatical material is kept verbatim and open-class material is replaced by a
UPOS placeholder, e.g.

    She decided to leave it .        (target = leave/VERB)   ->  "to {SLOT} PRON"
    The problem was serious .        (target = problem/NOUN)  ->  "the {SLOT} was"
    The problem was serious .        (target = serious/ADJ)   ->  "was {SLOT} ."

The hybrid representation is the point: it keeps the cues that actually select a
lexical category while discarding the lexical semantics that would otherwise
make a slot easy for reasons unrelated to syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Set, Tuple

from .ud_io import BOS, EOS, Sentence, Token

SLOT = "{SLOT}"

#: The categories we build slots for.
TARGET_POS: Tuple[str, ...] = ("NOUN", "VERB", "ADJ")

#: Closed-class categories whose word forms are kept verbatim in a signature.
#: These are the grammatical cues -- determiners, prepositions, auxiliaries,
#: infinitival "to", complementizers, coordinators, punctuation.
DEFAULT_LEXICALIZED_UPOS: Tuple[str, ...] = (
    "DET",
    "ADP",
    "AUX",
    "PART",
    "SCONJ",
    "CCONJ",
    "PUNCT",
)

#: Categories collapsed to a UPOS placeholder. PRON is abstracted rather than
#: lexicalized so that "to leave it / to tell them / to ask her" collapse into a
#: single frame; in a 250k-token treebank that difference decides whether a
#: frame clears the frequency threshold at all.
DEFAULT_ABSTRACTED_UPOS: Tuple[str, ...] = (
    "PRON",
    "NOUN",
    "PROPN",
    "VERB",
    "ADJ",
    "ADV",
    "NUM",
    "INTJ",
    "SYM",
    "X",
)


@dataclass(frozen=True)
class Occurrence:
    """One attested instantiation of one signature."""

    sent_key: Tuple[str, int]  # (corpus source, sentence index)
    target_idx: int
    upos: str
    lemma: str
    deprel: str
    feats: str


class SignatureEncoder:
    """Maps context tokens to signature elements under a fixed policy."""

    def __init__(
        self,
        lexicalized_upos: Sequence[str] = DEFAULT_LEXICALIZED_UPOS,
        abstracted_upos: Sequence[str] = DEFAULT_ABSTRACTED_UPOS,
        lexicalized_adverbs: Set[str] | None = None,
    ) -> None:
        self.lexicalized_upos = set(lexicalized_upos)
        self.abstracted_upos = set(abstracted_upos)
        self.lexicalized_adverbs = lexicalized_adverbs or set()

    def encode(self, tok: Token) -> str:
        if tok.upos == "ADV" and tok.form.lower() in self.lexicalized_adverbs:
            return tok.form.lower()
        if tok.upos in self.lexicalized_upos:
            return tok.form.lower()
        if tok.upos in self.abstracted_upos:
            return tok.upos
        return tok.upos

    def is_abstract(self, element: str) -> bool:
        return element in self.abstracted_upos or element in {BOS, EOS}

    def element_kind(self, element: str) -> str:
        """Classify one signature element.

        ``lexical`` elements are the grammatical cues -- "to", "the", "very",
        "will" -- that do the category-selecting work in a test frame.
        ``abstract`` elements are UPOS placeholders standing in for open-class
        material, and ``punct`` elements carry no continuation to measure.
        """
        if element == SLOT:
            return "slot"
        if element in {BOS, EOS}:
            return "boundary"
        if element in self.abstracted_upos:
            return "abstract"
        if not any(ch.isalnum() for ch in element):
            return "punct"
        return "lexical"


def _context_element(
    sent: Sentence, idx: int, encoder: SignatureEncoder
) -> str:
    if idx < 0:
        return BOS
    if idx >= len(sent.tokens):
        return EOS
    return encoder.encode(sent.tokens[idx])


def windows(max_left: int, min_right: int, max_right: int, min_total: int = 2):
    """Enumerate the (left, right) window sizes to extract.

    ``min_right >= 1`` is not cosmetic. Task 2 scores the surprisal of the
    tokens *following* the slot, so a frame with no right context would leave
    the measured continuation unattested -- reintroducing exactly the invented
    material the pipeline is meant to eliminate.
    """
    out = []
    for left in range(0, max_left + 1):
        for right in range(min_right, max_right + 1):
            if left + right >= min_total:
                out.append((left, right))
    return out


def signature_for(
    sent: Sentence,
    target_idx: int,
    left: int,
    right: int,
    encoder: SignatureEncoder,
) -> str:
    parts: List[str] = []
    for off in range(-left, 0):
        parts.append(_context_element(sent, target_idx + off, encoder))
    parts.append(SLOT)
    for off in range(1, right + 1):
        parts.append(_context_element(sent, target_idx + off, encoder))
    return " ".join(parts)


def iter_target_tokens(sent: Sentence) -> Iterator[Token]:
    for tok in sent.tokens:
        if tok.upos in TARGET_POS:
            yield tok


def signature_shape(signature: str) -> Tuple[int, int]:
    """(n_left, n_right) for a signature string."""
    parts = signature.split(" ")
    i = parts.index(SLOT)
    return i, len(parts) - i - 1
