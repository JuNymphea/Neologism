"""Tagger-based checks on probe items.

A probe item is a *fragment*, not a sentence: ``The {NEOLOGISM} of`` ends in a
preposition with no object, by design, because "of" is exactly the continuation
whose surprisal we want to score. So the sentence-level checks used when this
pipeline still built full sentences (single verbal ROOT, no dangling
preposition, terminal punctuation) do not apply and would reject correct items.

What remains meaningful on a fragment:

* the tagger must place a real word of the target category in the slot, and
  not place words of the other two categories there -- this is the cheap
  stand-in for the control-word calibration of Section 4.3;
* the fragment must not contain material the tagger cannot analyse at all.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

NEOLOGISM = "{NEOLOGISM}"

_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ll",
                    "'ve", "'d", "'m"}


def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        out = tok if i == 0 else out + ("" if tok in _NO_SPACE_BEFORE else " ") + tok
    return out


class TaggerCheck:
    """Does substituting real words of a category make a tagger see it there?"""

    #: Canonical, uncontroversial environment per category. A probe word the
    #: tagger cannot place even here is not a usable probe.
    SCREENING_FRAMES = {
        "NOUN": ("They", "saw", "the", NEOLOGISM, "at", "the", "time", "."),
        "VERB": ("They", "will", NEOLOGISM, "it", "at", "the", "time", "."),
        "ADJ": ("It", "was", "very", NEOLOGISM, "at", "the", "time", "."),
    }

    def __init__(self, nlp, probes: Dict[str, List[str]], n_probes: int = 12,
                 screen: bool = True):
        self.nlp = nlp
        self.n_probes = n_probes
        self.probes = self._screen(probes) if screen else probes

    # -- probe hygiene -------------------------------------------------------

    def _screen(self, probes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Drop probe words the tagger will not place in a canonical frame.

        The supplied lists are sampled from the model's tokenizer and contain
        proper nouns ("anna", "australia"), which a tagger labels PROPN rather
        than NOUN. Without this screen a perfectly good noun frame is penalised
        for the probe list's contents rather than its own behaviour.
        """
        kept: Dict[str, List[str]] = {}
        for pos, words in probes.items():
            frame = self.SCREENING_FRAMES.get(pos)
            if frame is None:
                kept[pos] = list(words)
                continue
            texts = [self._render(frame, w) for w in words]
            good = []
            for word, doc in zip(words, self.nlp.pipe(texts, batch_size=64)):
                for tok in doc:
                    if tok.text.lower() == word.lower():
                        if tok.pos_ == pos:
                            good.append(word)
                        break
            kept[pos] = good
        return kept

    # -- rendering -----------------------------------------------------------

    @staticmethod
    def _render(tokens: Sequence[str], word: str) -> str:
        """Substitute the probe and present the item as it will be shown.

        Capitalization matters: an uncapitalized sentence-initial "the" costs
        the tagger accuracy, which silently penalised every item whose frame
        already begins the string. The first token is left alone when it is the
        slot itself, so a probe is never turned into a proper noun.
        """
        text = detokenize([word if t == NEOLOGISM else t for t in tokens])
        if tokens and tokens[0] != NEOLOGISM and text:
            text = text[0].upper() + text[1:]
        return text

    # -- checks --------------------------------------------------------------

    def accuracy(self, tokens: Sequence[str], pos: str) -> float:
        words = self.probes.get(pos, [])[: self.n_probes]
        if not words:
            return 0.0
        hits = 0
        texts = [self._render(tokens, w) for w in words]
        for word, doc in zip(words, self.nlp.pipe(texts, batch_size=32)):
            for tok in doc:
                if tok.text.lower() == word.lower():
                    hits += tok.pos_ == pos
                    break
        return hits / len(words)

    #: Subject / finite-auxiliary pairs that do not agree.
    _BAD_AGREEMENT = {
        ("they", "is"), ("they", "was"), ("they", "has"), ("they", "does"),
        ("it", "are"), ("it", "were"), ("it", "have"), ("it", "do"),
        ("someone", "are"), ("someone", "were"), ("someone", "have"),
    }

    def is_wellformed(self, tokens: Sequence[str], pos: str,
                      closure: Sequence[str] = ()) -> bool:
        """Structural check on the item, completed into a sentence.

        A probe item is a fragment by design -- "The {NEOLOGISM} of" ends in the
        preposition whose surprisal we score. A parser cannot judge that, so the
        item is temporarily completed with `closure` and the *sentence* is
        checked. The closure is never scored and never shown; it exists only so
        that the structural checks have something well-formed to look at.
        """
        for a, b in zip(tokens, tokens[1:]):
            if a.lower() == b.lower() and a != NEOLOGISM:
                return False  # "one one", "the the"
            if (a.lower(), b.lower()) in self._BAD_AGREEMENT:
                return False  # "They is", "It are"

        full = list(tokens) + list(closure)
        for probe in self.probes.get(pos, ["thing"])[:3]:
            doc = self.nlp(self._render(full, probe))
            roots = [t for t in doc if t.dep_ == "ROOT"]
            if len(roots) != 1 or roots[0].pos_ not in {"VERB", "AUX"}:
                return False
            if any(t.dep_ == "dep" for t in doc):
                return False
            if not any(t.dep_ in {"nsubj", "nsubjpass", "expl", "csubj"}
                       for t in doc):
                return False
            for t in doc:
                if t.pos_ == "ADP" and not any(
                    c.dep_ in {"pobj", "pcomp", "obj"} for c in t.children
                ):
                    return False
                if t.pos_ == "DET" and t.head.pos_ not in {
                    "NOUN", "PROPN", "PRON", "NUM", "ADJ"
                }:
                    return False
                if (t.pos_ == "PRON" and t.dep_ == "appos"
                        and t.head.pos_ in {"NOUN", "PROPN"}):
                    return False
                is_np_head = (t.pos_ == "NOUN" and t.tag_ == "NN") or (
                    t.lower_ == "one" and t.pos_ in {"NUM", "NOUN", "PRON"})
                in_arg = t.dep_ in {"attr", "dobj", "pobj", "nsubj", "conj",
                                    "nsubjpass"}
                has_det = any(c.dep_ in {"det", "poss", "nummod"}
                              for c in t.children)
                if is_np_head and in_arg and not has_det:
                    return False
        return True


def load_tagger():
    """spaCy's small English model, with unused components disabled."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
