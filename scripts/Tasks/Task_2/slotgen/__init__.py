"""Corpus-derived syntactic slot induction for Task 2.

Replaces hand-written POS-selective slots with local syntactic frames induced
from a POS-annotated treebank (UD English EWT), realized as minimal probe
items, and validated on held-out and cross-corpus data.

    ud_io     read CoNLL-U; corpus statistics used as naturalness evidence
    frames    steps 1-2: tokens -> hybrid context signatures
    induce    steps 3-6: count, threshold, score, de-duplicate by construction
    minimal   step 7: frame -> minimal prefix + diagnostic continuation
    tagging   tagger checks on probe items
    validate  re-estimate selectivity on held-out and external corpora
    overrides human accept / rewrite / reject of individual items
"""

from . import frames, induce, minimal, overrides, tagging, ud_io, validate  # noqa: F401

__all__ = ["ud_io", "frames", "induce", "minimal", "tagging", "validate",
           "overrides"]
