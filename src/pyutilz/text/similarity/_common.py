"""Shared-imports hub, tunable constants and small validation/coverage helpers for the ``pyutilz.text.similarity`` submodules."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import unicodedata
import logging

# Pin the logger name to the historical flat-module path so log records keep the same channel after the split into a subpackage.
logger = logging.getLogger("pyutilz.text.similarity")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


from jellyfish import levenshtein_distance  # noqa: F401 -- re-exported for ``sentences.py``/``basics.py``
from pyutilz.text.strings import strip_doubled_characters  # noqa: F401 -- re-exported for ``basics.py``
from typing import Optional  # re-exported for every submodule

# Above this many tokens per sentence, the O(w_min * N_a * N_b) greedy matching pass
# (a full rescan of the similarity matrix for each of the w_min greedy picks, on top of the
# O(N_a * N_b) matrix fill) starts to dominate. Benchmarked: N=10 -> ~0.3ms, N=80 -> ~25-30ms
# per call (pure-Python path). Not severe enough to justify a heap-based rewrite (which would
# need a differential-correctness test against the current tie-breaking to avoid behavior
# drift), but callers feeding longer inputs should be aware of the quadratic-ish cost.
SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD = 50

_VALID_COVERAGE_SIDES = ("min", "max", "both")

def _validate_coverage_side(coverage_side: str) -> None:
    """Raise ValueError unless `coverage_side` is one of `_VALID_COVERAGE_SIDES` ("min"/"max"/"both")."""
    if coverage_side not in _VALID_COVERAGE_SIDES:
        raise ValueError(f"coverage_side must be one of {_VALID_COVERAGE_SIDES}, got {coverage_side!r}")


def _check_word_coverage(
    words: list,
    matched_sims,
    min_word_similarity: float,
    required_coverage: Optional[float],
    required_matched_words: Optional[int],
    stop_set: set,
) -> bool:
    """True iff enough of `words`' non-stopword entries have an achieved match similarity
    (`matched_sims`, same length/order as `words`, 0.0 where a word went unmatched - always true for
    every word on the w_max side of a greedy w_min-pair match) at or above `min_word_similarity`.

    A word not in `stop_set` is "content"; coverage is judged only over content words - a sentence
    made entirely of stopwords (or empty) trivially passes, since there is nothing to require
    coverage of. Both `required_matched_words` (absolute count) and `required_coverage` (fraction of
    content words) may be supplied together; both must pass.
    """
    content_indices = [i for i, w in enumerate(words) if w not in stop_set]
    if not content_indices:
        return True
    covered = sum(1 for i in content_indices if matched_sims[i] >= min_word_similarity)
    if required_matched_words is not None and covered < required_matched_words:
        return False
    if required_coverage is not None and covered / len(content_indices) < required_coverage:
        return False
    return True


def _nfc_words(words: list) -> list:
    """`words` with every entry NFC-normalized (composed form).

    The pure-Python path measures distance with ``jellyfish.levenshtein_distance``, whose Rust
    implementation counts a base character plus its combining marks as ONE unit, while the numba
    kernel operates on raw utf-32 codepoints and is codepoint-exact. Without a shared
    normalization, an NFD input (routine from PDF extraction and macOS filenames) scored
    differently depending only on whether numba happened to be installed -- e.g.
    ``["cafe\u0301"]`` vs ``["caf\u00e9"]`` gave 0.775 pure-Python and 0.675 under numba.
    Normalizing BOTH sides to NFC makes the two paths see identical strings.
    """
    return [unicodedata.normalize("NFC", w) if isinstance(w, str) else w for w in words]


def _strip_stop_words(words: list, stop_words: Optional[list]) -> list:
    """`words` with every entry in `stop_words` removed, or `words` unchanged when that would empty the
    list (a sentence made entirely of stop words has nothing else to score against) or when `stop_words`
    is falsy.

    Applied BEFORE the w_max length-normalization (2026-08-12's own fix) sees either sentence, not only
    inside the opt-in coverage gate `_check_word_coverage` already excludes stop words from: a connective
    word ("some", "I", "have") still counted as query LENGTH `w_max` couldn't explain, silently dragging a
    correct single-word match below any reasonable score threshold - measured live in autopsia's complaint
    parser, 2026-08-16 ("some nausea" scored 0.0 against "Nausea", "I have nausea" scored 0.56, both used
    to score 1.0 before the w_max fix). `stop_words` was already a real, threaded parameter every call site
    below accepts - this widens what it does (base score, not just the coverage gate), it does not add new
    API surface."""
    if not stop_words:
        return words
    stop_set = set(stop_words)
    filtered = [w for w in words if w not in stop_set]
    if not filtered:
        return words
    return filtered
