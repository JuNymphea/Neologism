"""Minimal-frame realization and diagnostic continuations.

Task 2 is a **corpus-calibrated syntactic compatibility probe**. It asks one
question:

    When a neologism is placed in a local syntactic environment that, in real
    text, strongly selects one part of speech, does its continuation behaviour
    look more like known words of that POS than of the other two?

Two consequences follow, and they are what this module implements.

**1. A probe item is a minimal grammatical prefix, not a sentence.**
The corpus frame ``to {SLOT} PRON`` becomes

    They tried to {NEOLOGISM} it

and stops there. Nothing is appended to reach a sentence boundary or a fixed
token count. Earlier versions padded every item out to four scored tokens with
tails like *one more time* / *at the time*; those tokens carried no syntactic
evidence about the slot, repeated across slots, and diluted the measurement.

**2. Surprisal is scored only over the continuation the frame licenses.**
Each frame ``f`` carries its own diagnostic continuation ``D_f`` -- the frame's
own right context, and nothing else:

    to {SLOT} PRON        ->  score  "it"
    very {SLOT} NOUN      ->  score  "thing"
    the {SLOT} of         ->  score  "of"
    the {SLOT} and NOUN   ->  score  "and something"

so ``|D_f|`` varies by construction rather than being fixed at n = 3 or 4.
Different constructions carry their syntactic evidence over different spans;
a fixed window has no theoretical justification and mixes in tokens that say
nothing about the slot.

    S(w, f) = (1/|D_f|) * sum_{t in D_f} -log P(t | prefix containing w)

Semantic bleaching applies only to material outside the frame, and the
vocabulary for it is tiny: no concept-specific cue should sit anywhere in the
item. Because no long sentence is assembled, this is now easy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .frames import SLOT, SignatureEncoder
from .induce import FrameStats
from .ud_io import BOS, EOS

NEOLOGISM = "{NEOLOGISM}"

# ---------------------------------------------------------------------------
# The entire generic vocabulary available outside the corpus frame.
# ---------------------------------------------------------------------------

#: Minimal left context, keyed on the class of the frame's left edge. Each entry
#: adds only what the frame's own leftmost element requires in order to be
#: grammatical -- "to" needs a governing verb, "n't" needs an auxiliary.
OPENERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "START":  ((),),
    "TO":     (("They", "tried"), ("Someone", "tried")),
    "NEG":    (("They", "do"), ("Someone", "does")),
    "AUX":    (("They",), ("It",), ("Someone",)),
    "CLITIC": (("It",), ("Someone",)),
    "DET":    ((), ("They", "have", "seen")),
    "ADP":    (("They", "spoke"), ("Someone", "spoke")),
    "DEG":    (("It", "was"), ("It", "was", "a"), ("They", "were")),
    "WH":     (("They", "know"), ("Someone", "knows")),
    "CCONJ":  (("They", "did", "it"),),
    "COMMA":  (("It", "was"),),
    "NOUN":   (("The",), ("They", "have", "seen", "the")),
    "ADJ":    (("The",), ("They", "have", "seen", "the")),
    "VERB":   (("They",), ("Someone",)),
    "PRON":   ((),),
    "OTHER":  ((), ("They",), ("It", "was")),
}

#: Generic filler pools. A pool, not a single word per UPOS: a filler has to
#: fit the *syntactic position*, not merely carry the right tag. "other" is an
#: adjective but is attributive-only, so "very {SLOT} and other" is bad English
#: even though a tagger accepts it. The choice among a pool is made by checking
#: the corpus for whether that word is attested in the position's own relation.
FILLERS: Dict[str, Tuple[str, ...]] = {
    "PRON": ("it", "them"),
    "NOUN": ("thing", "something", "one"),
    "PROPN": ("someone",),
    "VERB": ("happened", "did", "started", "changed"),
    "ADJ": ("different", "similar", "certain", "other", "same"),
    "ADV": ("again", "then"),
    "NUM": ("one",),
    "INTJ": ("well",),
    "SYM": ("one",),
    "X": ("thing",),
}

EDGE_CLASS: Dict[str, str] = {
    "to": "TO",
    "n't": "NEG", "not": "NEG",
    "'s": "CLITIC",
    "and": "CCONJ", "or": "CCONJ", "but": "CCONJ",
    "very": "DEG", "more": "DEG", "most": "DEG", "so": "DEG", "too": "DEG",
    "quite": "DEG", "really": "DEG", "as": "DEG",
    "how": "WH", "what": "WH", "why": "WH",
    ",": "COMMA",
}
_DETS = {"the", "a", "an", "this", "that", "these", "those", "some", "no",
         "his", "her", "their", "our", "its", "my", "your"}
_AUXES = {"will", "can", "would", "could", "should", "may", "might", "must",
          "is", "are", "was", "were", "be", "been", "am", "has", "have", "had",
          "do", "does", "did"}
_PREPS = {"of", "in", "on", "for", "with", "at", "by", "from", "about", "into",
          "over", "under", "through", "than", "to"}
_SINGULAR_AUX = {"is", "was", "has", "does"}
_PLURAL_AUX = {"are", "were", "have", "do"}
_DEGREE = {"very", "more", "most", "so", "too", "quite", "really", "as"}
_INFINITIVE_TRIGGERS = {"to", "will", "can", "would", "could", "should", "do",
                        "does", "did", "n't", "not", "may", "might", "must"}
_PLURAL_TRIGGERS = {"are", "were"}


#: Minimal completions used *only* to give the structural checks a well-formed
#: sentence to inspect. Never scored, never emitted.
COMPLETIONS: Dict[str, Tuple[str, ...]] = {
    "TO": ("do", "it"),
    "ADP": ("it",),
    "DET": ("thing",),
    "CCONJ": ("something",),
    "AUX": ("there",),
    "DEG": ("thing",),
    "NEG": ("do", "it"),
    "COMMA": ("something",),
    "WH": ("it", "is"),
    "NOUN": (), "PRON": (), "VERB": (), "ADJ": ("thing",),
    "CLITIC": ("thing",), "START": (), "OTHER": (),
}

_FINITE_FORMS = {"is", "was", "are", "were", "has", "have", "had", "do",
                 "does", "did", "can", "will", "would", "could", "should",
                 "tried", "seen", "spoke", "know", "knows", "happened", "be"}


def closure_for(item_tokens: Sequence[str], last_edge: str) -> Tuple[str, ...]:
    """Material appended *only* so the structural check sees a full sentence.

    Never scored and never shown. A predicate is added when the item has no
    finite verb of its own, which is what lets a bare "The {SLOT} of" be
    validated without padding the item itself.
    """
    out = list(COMPLETIONS.get(last_edge, ()))
    has_verb = any(t.lower() in _FINITE_FORMS for t in item_tokens)
    if not has_verb:
        out += ["was", "there"]
    return tuple(out + ["."])


def edge_class(element: str, encoder: SignatureEncoder) -> str:
    kind = encoder.element_kind(element)
    if kind == "abstract":
        return element if element in OPENERS else "OTHER"
    low = element.lower()
    if low in EDGE_CLASS:
        return EDGE_CLASS[low]
    if low in _DETS:
        return "DET"
    if low in _AUXES:
        return "AUX"
    if low in _PREPS:
        return "ADP"
    if kind == "punct":
        return "COMMA"
    return "OTHER"


def _fits_position(word: str, upos: str, deprel: str, dists: dict | None) -> bool:
    """Is `word` attested in the corpus bearing `deprel` as `upos`?

    This is what separates "has the right tag" from "occurs in this position".
    """
    if dists is None or deprel in ("_", ""):
        return True
    attested = dists.get(word.lower())
    if not attested:
        return False
    if (upos, deprel) in attested:
        return True
    # Allow a coarser match on the relation's main label ("conj:preconj" etc).
    base = deprel.split(":")[0]
    return any(u == upos and d.split(":")[0] == base for u, d in attested)


def _fill(element: str, prev: str | None, nxt: str | None,
          clause_initial: bool, variant: int = 0,
          deprel: str = "_", dists: dict | None = None,
          target_feats: str = "") -> str:
    if element == "VERB":
        if (nxt or "").lower() == "to":
            return "tried"
        if (prev or "").lower() in _INFINITIVE_TRIGGERS:
            return "do"
        # An object follows: the filler has to be transitive ("happened the
        # thing" is not English).
        if (nxt or "").lower() in _DETS:
            return "did"
        options = FILLERS["VERB"]
    elif element == "NOUN":
        if (nxt or "").lower() in _PLURAL_TRIGGERS:
            return "things"
        options = FILLERS["NOUN"]
        # "something" is determiner-like and cannot head an NP that already has
        # a determiner or an attributive modifier.
        if (prev or "").lower() in _DETS | _DEGREE | {"other", "same"}:
            options = tuple(o for o in options if o != "something") or ("thing",)
    elif element == "PRON":
        if clause_initial:
            # The subject has to agree both with a following auxiliary and
            # with the person/number the frame's own corpus occurrences show;
            # otherwise "They {SLOT} to" would demand a 1sg control form.
            if (nxt or "").lower() in _SINGULAR_AUX:
                return "it"
            if "Person=1" in target_feats and "Number=Sing" in target_feats:
                return "I"
            if "Person=3" in target_feats and "Number=Sing" in target_feats:
                return "it"
            return "they"
        return "it"
    else:
        options = FILLERS.get(element, ("thing",))
        if element == "ADJ" and (prev or "").lower() in {"a", "an"}:
            options = tuple(o for o in options if o != "same") or ("other",)

    # Keep only fillers actually attested in this syntactic position, then
    # rotate through what survives.
    fitting = tuple(o for o in options
                    if _fits_position(o, element, deprel, dists))
    options = fitting or options
    return options[variant % len(options)]


# ---------------------------------------------------------------------------
# Construction disjointness from the training templates
# ---------------------------------------------------------------------------

#: The templates the neologism embedding is *trained* in (paper Section 3.2).
TRAINING_TEMPLATES: Dict[str, str] = {
    "unbiased": "Make your answer reflect the following word: {NEOLOGISM}.",
    "verb": "Please {NEOLOGISM} your answer.",
    "noun": "Please answer this question with a {NEOLOGISM}.",
    "adjective": "Your answer should be as {NEOLOGISM} as possible.",
}

#: Local constructions those templates instantiate, as (left cue, right cue).
#: ``None`` is a wildcard and ``""`` means "nothing there". A probe frame that
#: matches one of these is excluded: scoring a neologism in the construction it
#: was trained in measures memorisation of that construction, not generalisation
#: about lexical category.
TRAINING_CONSTRUCTIONS: Tuple[Tuple[str | None, str | None], ...] = (
    ("as", "as"),        # adjective: "as {NEOLOGISM} as possible"
    ("a", ""),           # noun: "with a {NEOLOGISM}." -- slot phrase-final
    ("with", None),      # noun: the governing preposition
    (None, "your"),      # verb: "Please {NEOLOGISM} your answer"
    (":", None),         # unbiased: "the following word: {NEOLOGISM}"
)


def collides_with_training(signature: str, encoder: SignatureEncoder) -> str | None:
    """Name the training construction a frame duplicates, if any."""
    els = signature.split(" ")
    i = els.index(SLOT)
    left = [e for e in els[:i] if e not in (BOS, EOS)]
    right = [e for e in els[i + 1:] if e not in (BOS, EOS)]

    def cue(seq: List[str], from_right: bool) -> str | None:
        """The adjacent lexical cue: "" if nothing is there, None if abstract."""
        if not seq:
            return ""
        el = seq[-1] if from_right else seq[0]
        return None if encoder.element_kind(el) == "abstract" else el.lower()

    lcue = cue(left, from_right=True)      # element immediately left of slot
    rcue = cue(right, from_right=False)    # element immediately right of slot
    for want_l, want_r in TRAINING_CONSTRUCTIONS:
        if want_l is not None and lcue != want_l:
            continue
        if want_r is not None and rcue != want_r:
            continue
        return f"{want_l or '*'} {{SLOT}} {want_r or '*'}"
    return None


# ---------------------------------------------------------------------------


@dataclass
class ProbeItem:
    """One evaluation item: a prefix, and the tokens whose surprisal is scored."""

    signature: str
    pos: str
    prefix_tokens: List[str]      # everything up to and including the slot
    diagnostic: List[str]         # D_f, the frame's own licensed continuation
    morph_requirement: str        # what a control word here has to look like
    opener: Tuple[str, ...]
    text: str                     # prefix + diagnostic, for reading
    tagger_accuracy: float
    tagger_accuracy_other: Dict[str, float]
    tagger_margin: float

    def prefix_for(self, word: str) -> str:
        """The string the model is conditioned on, with `word` in the slot.

        The indefinite article is adjusted to the substituted word: a control
        noun like "idea" in a frame containing "a {SLOT}" would otherwise give
        "a idea" and manufacture surprisal that has nothing to do with the
        word's category.
        """
        toks = [word if t == NEOLOGISM else t for t in self.prefix_tokens]
        for i, tok in enumerate(toks[:-1]):
            if tok.lower() in {"a", "an"}:
                vowel = toks[i + 1][:1].lower() in set("aeiou")
                fixed = "an" if vowel else "a"
                toks[i] = fixed.capitalize() if tok[0].isupper() else fixed
        return _capitalize(detokenize(toks))

    def as_dict(self) -> dict:
        d = dict(vars(self))
        d["n_diagnostic"] = len(self.diagnostic)
        return d


_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ll",
                    "'ve", "'d", "'m"}


def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        out = tok if i == 0 else out + ("" if tok in _NO_SPACE_BEFORE else " ") + tok
    return out


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text and not text.startswith(NEOLOGISM) else text


def build(
    fs: FrameStats,
    encoder: SignatureEncoder,
    opener: Tuple[str, ...],
    variant: int,
    dists: dict | None = None,
    bigrams=None,
    min_bigram: int = 1,
) -> Tuple[List[str], List[str]] | None:
    """(prefix_tokens, diagnostic) for one opener/filler choice, or None."""
    els = fs.elements()
    i = els.index(SLOT)
    left_raw, right_raw = els[:i], els[i + 1:]

    if BOS in left_raw and opener:
        return None
    left_raw = [e for e in left_raw if e != BOS]
    right_raw = [e for e in right_raw if e not in (EOS,)]
    # Punctuation carries no syntactic evidence about the slot, so it is never
    # part of the diagnostic continuation.
    right_raw = [e for e in right_raw
                 if encoder.element_kind(e) != "punct"]
    if not right_raw:
        return None
    # A clitic ("'ve", "'s", "n't") is not usable evidence about the slot.
    if any(e.startswith("'") for e in right_raw):
        return None

    def render(seq, prev0, clause_start, offsets):
        out: List[str] = []
        for j, el in enumerate(seq):
            prev = out[-1] if out else prev0
            nxt = seq[j + 1] if j + 1 < len(seq) else None
            if encoder.element_kind(el) == "abstract":
                out.append(_fill(el, prev, nxt, clause_start and not out,
                                 variant, fs.context_deprel_at(offsets[j]),
                                 dists, fs.dominant_feats))
            else:
                out.append(el)
        return out

    left = render(left_raw, opener[-1] if opener else None, not opener,
                  list(range(-len(left_raw), 0)))
    diagnostic = render(right_raw, left[-1] if left else
                        (opener[-1] if opener else None), False,
                        list(range(1, len(right_raw) + 1)))

    joined = list(left) + diagnostic
    for a, b in zip(joined, joined[1:]):
        if b in {"same", "one"} and a.lower() in {"a", "an"}:
            return None
        if a.lower() == "a" and b[:1].lower() in set("aeiou"):
            return None  # would need "an"

    prefix = list(opener) + left + [NEOLOGISM]

    # The context up to the neologism must itself be a natural English prefix,
    # and prefix + D_f must continue naturally. A defect *before* the slot
    # contaminates the whole conditional context, so this is checked on the
    # corpus rather than left to the tagger: "they tried to" is attested
    # thousands of times, "the same to" essentially never.
    if bigrams is not None:
        span = list(opener) + left + diagnostic  # slot excluded
        boundary = len(opener) + len(left)       # bigrams must not cross it
        for k, (a, b) in enumerate(zip(span, span[1:])):
            if k + 1 == boundary:
                continue
            if bigrams[(a.lower(), b.lower())] < min_bigram:
                return None

    return prefix, diagnostic


def morph_requirement(fs: FrameStats) -> str:
    """The inflection a word must have to sit in this frame.

    Read off the frame's own annotation: the dominant morphological features of
    the words the corpus actually puts here. A control word for
    ``They {SLOT} it`` has to be a present-tense form agreeing with a plural
    subject; one for ``A {SLOT} to`` has to be a singular count noun. Without
    this, control-word surprisal reflects inflection rather than category.
    """
    feats = fs.dominant_feats
    return feats if feats and feats != "_" else "Any"


def make_item(
    fs: FrameStats,
    encoder: SignatureEncoder,
    check,
    min_accuracy: float = 0.85,
    min_margin: float = 0.10,
    dists: dict | None = None,
    bigrams=None,
    min_bigram: int = 1,
) -> ProbeItem | None:
    """Best minimal realization of `fs`, or None if it has none.

    Candidates are checked in a fixed order -- grammaticality, then tagger
    consistency -- and the most discriminating one is returned. `None` means
    the frame cannot be realized neutrally and should be replaced by the next
    corpus candidate.
    """
    pos = fs.dominant_pos
    els = fs.elements()
    i = els.index(SLOT)
    left_raw = els[:i]
    left_edge = "START" if (not left_raw or left_raw[0] == BOS) else edge_class(
        left_raw[0], encoder)
    openers = OPENERS.get(left_edge, OPENERS["OTHER"])
    if left_edge == "AUX" and left_raw:
        aux = left_raw[0].lower()
        if aux in _SINGULAR_AUX:
            openers = (("It",),)
        elif aux in _PLURAL_AUX:
            openers = (("They",),)

    best: ProbeItem | None = None
    best_key: Tuple = ()
    n_variants = max(len(v) for v in FILLERS.values())
    for oi, opener in enumerate(openers):
        for variant in range(n_variants):
            built = build(fs, encoder, opener, variant, dists, bigrams,
                          min_bigram)
            if built is None:
                continue
            prefix, diagnostic = built
            tokens = prefix + diagnostic
            closure = closure_for(tokens, edge_class(diagnostic[-1], encoder))
            if not check.is_wellformed(tokens, pos, closure):
                continue
            acc = check.accuracy(tokens, pos)
            if acc < min_accuracy:
                continue
            others = {p: check.accuracy(tokens, p)
                      for p in ("NOUN", "VERB", "ADJ") if p != pos}
            margin = acc - max(others.values())
            if margin < min_margin:
                continue
            key = (round(margin, 1), round(acc, 1), -len(opener), -oi,
                   -variant, -len(tokens))
            if best is None or key > best_key:
                best_key = key
                best = ProbeItem(
                    signature=fs.signature,
                    pos=pos,
                    morph_requirement=morph_requirement(fs),
                    prefix_tokens=([_capitalize(prefix[0])] + prefix[1:]
                                   if prefix and prefix[0] != NEOLOGISM else prefix),
                    diagnostic=diagnostic,
                    opener=opener,
                    text=_capitalize(detokenize(tokens)),
                    tagger_accuracy=round(acc, 3),
                    tagger_accuracy_other={k: round(v, 3) for k, v in others.items()},
                    tagger_margin=round(margin, 3),
                )
    return best
