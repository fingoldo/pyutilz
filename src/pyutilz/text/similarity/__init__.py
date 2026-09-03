"""``pyutilz.text.similarity`` -- string and sentence similarity metrics.

Historically a single flat module; split into a subpackage of cohesive submodules
(``basics``, ``sentences``, ``_numba_kernels``, ``_numba_api``, plus the shared
``_common`` hub) once it grew past the 1000-LOC limit. This ``__init__``
re-exports the entire original module namespace so that
``from pyutilz.text.similarity import X`` and the lazy ``pyutilz.similarity.X``
alias keep resolving unchanged -- including the private helpers and numba kernels
the test suite imports directly.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Shared module-level names (imports/constants) preserved from the flat module
# ----------------------------------------------------------------------------------------------------------------------------

from ._common import (
    annotations,
    logging,
    logger,
    levenshtein_distance,
    strip_doubled_characters,
    Optional,
    SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD,
    _VALID_COVERAGE_SIDES,
    _validate_coverage_side,
    _check_word_coverage,
    _nfc_words,
    _strip_stop_words,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Public functions/classes, re-exported from the cohesive submodules
# ----------------------------------------------------------------------------------------------------------------------------

from .basics import levenshtein_strings_similarity, contigous_strings_similarity, normalize_sentence
from .sentences import sentences_similarity
from ._numba_api import (
    HAS_NUMBA,
    SentenceSimilarityIndex,
    pack_sentence,
    sentences_similarity_numba,
    sentences_similarity_numba_batch,
    sentences_similarity_numba_packed,
)

# ----------------------------------------------------------------------------------------------------------------------------
# The numba kernels and their Python-side packing/dispatch helpers only exist when numba imported
# successfully; mirror the flat module, which bound these names inside the same conditional block.
# ----------------------------------------------------------------------------------------------------------------------------

if HAS_NUMBA:
    from ._numba_kernels import (
        nb,
        np,
        _SORTED_MATCH_THRESHOLD,
        _lev_dist_flat,
        _sentences_similarity_core,
        _greedy_match_sorted,
        _fill_sim_matrix,
        _sentences_similarity_core_sorted,
        _greedy_match_with_tracking,
        _sentences_similarity_core_with_matches,
        _pack_words,
        _run_sentences_similarity_core,
        _compare_one_candidate,
        _fill_sim_matrix_candidate,
        _compare_one_candidate_with_matches,
        _sentences_similarity_batch_core,
        _sentences_similarity_batch_parallel,
        _sentences_similarity_batch_with_matches_core,
        _sentences_similarity_batch_with_matches_parallel,
    )
    from ._numba_api import _prepare_batch

# ----------------------------------------------------------------------------------------------------------------------------
# Hide the submodule names so ``dir(pyutilz.text.similarity)`` matches the pre-split flat module
# exactly. Explicit `from . import` here (rather than relying on the implicit submodule binding the
# `from .X import name` statements above already produce) so mypy sees a real binding to delete
# instead of needing an unreliable ignore comment.
# ----------------------------------------------------------------------------------------------------------------------------

from . import _common, basics, sentences, _numba_kernels, _numba_api

# Deleting these bindings would make ``dir()`` depend on import ORDER rather than on a fixed
# property of the module: any later access through the import system (a
# ``mock.patch("<pkg>.<submodule>.name")`` target, a plain ``import <pkg>.<submodule>``
# anywhere in the process) re-sets the deleted attribute as a side effect, while code that
# runs before that sees an AttributeError. A ``__dir__`` hook (PEP 562) reports the same
# curated surface without ever removing the real attribute, so every access path stays
# reliable no matter what has been imported first -- the idiom pyutilz.system.system uses.
_SUBMODULE_NAMES = frozenset({"_common", "basics", "sentences", "_numba_kernels", "_numba_api"})


def __dir__() -> list:
    """Reports the facade's curated surface, hiding the submodule names the re-exports came from."""
    return sorted(n for n in globals() if n not in _SUBMODULE_NAMES)

# Explicit public surface: makes this facade's re-export intent self-documenting (rather than relying
# on nothing but "this is __init__.py") and gives the self-scan code-audit possibly_dead_import check
# an authoritative usage signal for names only ever consumed by external repos
# (``from pyutilz.text.similarity import X``), which its own-repo corpus scan can't see.
__all__ = [
    "annotations",
    "logging",
    "logger",
    "levenshtein_distance",
    "strip_doubled_characters",
    "Optional",
    "SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD",
    "_VALID_COVERAGE_SIDES",
    "_validate_coverage_side",
    "_check_word_coverage",
    "_nfc_words",
    "_strip_stop_words",
    "levenshtein_strings_similarity",
    "contigous_strings_similarity",
    "normalize_sentence",
    "sentences_similarity",
    "HAS_NUMBA",
    "SentenceSimilarityIndex",
    "pack_sentence",
    "sentences_similarity_numba",
    "sentences_similarity_numba_batch",
    "sentences_similarity_numba_packed",
]
if HAS_NUMBA:
    __all__ += [
        "nb",
        "np",
        "_SORTED_MATCH_THRESHOLD",
        "_lev_dist_flat",
        "_sentences_similarity_core",
        "_greedy_match_sorted",
        "_fill_sim_matrix",
        "_sentences_similarity_core_sorted",
        "_greedy_match_with_tracking",
        "_sentences_similarity_core_with_matches",
        "_pack_words",
        "_run_sentences_similarity_core",
        "_compare_one_candidate",
        "_fill_sim_matrix_candidate",
        "_compare_one_candidate_with_matches",
        "_sentences_similarity_batch_core",
        "_sentences_similarity_batch_parallel",
        "_sentences_similarity_batch_with_matches_core",
        "_sentences_similarity_batch_with_matches_parallel",
        "_prepare_batch",
    ]
