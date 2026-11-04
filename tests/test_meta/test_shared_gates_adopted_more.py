"""py-ci-shared gates of the 2026-11-04 round (pin e13229e), all at zero findings on the tree.

Each gate reads either ``src/pyutilz`` or ``tests/``; the floors sit well under the measured file counts (src/pyutilz about 270
parsed files, tests/ about 320) so a wrong root or an over-broad exclude fails instead of passing on nothing.
``caller_frame_mutation`` is wired advisory (owner decision): findings warn and never fail.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from py_ci_shared.caller_frame_mutation import assert_caller_frame_mutation
from py_ci_shared.cli_flag_spelling import assert_cli_flag_spelling
from py_ci_shared.definition_after_entry_guard import assert_definition_after_entry_guard
from py_ci_shared.duplicate_test_body import assert_no_duplicate_test_bodies
from py_ci_shared.group_blind_split import assert_group_blind_split
from py_ci_shared.loop_invariant_cast import assert_loop_invariant_cast
from py_ci_shared.ml_api_contracts import assert_ml_api_contracts
from py_ci_shared.njit_fastmath_nan import assert_njit_fastmath_nan
from py_ci_shared.njit_unsupported_calls import assert_njit_unsupported_calls
from py_ci_shared.oof_buffer_zero_init import assert_oof_buffer_zero_init
from py_ci_shared.optional_collaborator_wired import assert_optional_collaborator_wired
from py_ci_shared.ordinal_rank_scatter import assert_ordinal_rank_scatter
from py_ci_shared.state_before_work import assert_state_before_work
from py_ci_shared.systemexit_in_library import assert_no_systemexit_in_library
from py_ci_shared.truncating_discretisation import assert_truncating_discretisation
from py_ci_shared.unseeded_rng_fallback import assert_unseeded_rng_fallback
from py_ci_shared.verified_then_reopened import assert_verified_then_reopened

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "pyutilz"
TESTS = REPO_ROOT / "tests"


def test_no_duplicate_test_bodies():
    """A test copied under a new name and never edited asserts nothing new; a mutant on the branch its name promises survives."""
    assert_no_duplicate_test_bodies(TESTS, min_files=250)


def test_no_definition_after_the_entry_guard():
    """A def or class placed after `if __name__ == '__main__'` does not exist yet when the guard runs."""
    assert_definition_after_entry_guard(SRC, min_files=150)


def test_verified_file_is_not_reopened_by_path():
    """Verify a sidecar/hash, then open the same path again: the second open can see a file replaced in between."""
    assert_verified_then_reopened(SRC, min_files=150)


def test_stamp_is_not_written_before_the_work_it_gates():
    """A schedule stamp or counter written before the fallible work records progress that never happened."""
    assert_state_before_work(SRC, min_files=150)


def test_library_code_never_exits_the_process():
    """sys.exit / raise SystemExit in a library function kills worker threads and importers; raise a normal exception."""
    assert_no_systemexit_in_library(SRC, min_files=150)


def test_optional_collaborator_parameters_are_wired():
    """A *_memo / *_cache / *_fn parameter defaulting to None that no production call site passes is dead configuration."""
    assert_optional_collaborator_wired(SRC, min_files=150)


def test_cli_flag_stems_have_one_spelling():
    """The same flag registered as --a-b in one script and --a_b in another splits users' muscle memory."""
    assert_cli_flag_spelling(SRC, min_files=150)


def test_njit_fastmath_does_not_assume_finite_when_the_body_tests_for_nan():
    """fastmath=True lets LLVM fold isnan/isfinite to a constant; use the NaN-safe flag set where the body handles NaN."""
    assert_njit_fastmath_nan(SRC, min_files=150)


def test_njit_bodies_avoid_calls_numba_cannot_type():
    """typing.cast, np.median, warnings.warn and logger calls inside compiled bodies raise TypingError on another numba."""
    assert_njit_unsupported_calls(SRC, min_files=150)


def test_ordinal_rank_scatter_is_not_used_for_tied_scores():
    """ranks[argsort(x)] = arange(n) splits ties by input order and ranks NaN last."""
    assert_ordinal_rank_scatter(SRC, min_files=150)


def test_group_ids_reach_the_splitter():
    """A function that receives groups must not build a group-blind KFold / ShuffleSplit / train_test_split."""
    assert_group_blind_split(SRC, min_files=150)


def test_oof_buffers_are_not_zero_initialised_and_filled_per_fold():
    """An out-of-fold buffer filled from a splitter that may skip rows keeps zeros that read as predictions."""
    assert_oof_buffer_zero_init(SRC, min_files=150)


def test_ml_api_contracts():
    """best_iteration used as a count without +1, argmax compared to a raw label, an __init__ rewriting its own parameter."""
    assert_ml_api_contracts(SRC, min_files=150)


def test_integer_casts_are_not_guarded_only_by_a_distinct_count():
    """A few distinct values do not make a fractional target integral; astype(int) truncates it."""
    assert_truncating_discretisation(SRC, min_files=150)


def test_missing_seed_does_not_become_an_entropy_generator():
    """A function that turns seed=None into default_rng() / Random() makes every run differ; mark deliberate entropy."""
    assert_unseeded_rng_fallback(SRC, min_files=150)


def test_loop_invariant_casts_and_heavy_calls_are_hoisted():
    """A cast or heavy numpy call whose arguments never change across the loop belongs before it."""
    assert_loop_invariant_cast(SRC, min_files=150, heavy=True)


def test_dataframe_parameters_are_not_mutated_in_place():
    """X[col] = ..., X.columns = ... on a frame the caller passed in silently rewrites the caller's data.

    Advisory by owner decision: a finding warns (CallerFrameMutationWarning) and never fails; a broken walk still fails.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_caller_frame_mutation(SRC, advisory=True, min_files=150)
    for warning in caught:
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
