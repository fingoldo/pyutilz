"""py-ci-shared gates adopted with 1.17.0, all zero-tolerance: the tree has no finding for any of them, so a new one fails.

RNG seed range, sentinel settings read through `or` (the max_tokens=0 class), stale source citations, unlocked module caches,
swallowed I/O or broad failures, xfails that park fixable work, and import cycles. File floors sit well under the measured
counts (src/pyutilz about 270 parsed files, tests/ about 320).
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.atomic_write_staging import assert_atomic_write_staging
from py_ci_shared.hash_fed_by_array_copy import assert_no_hash_fed_by_array_copy
from py_ci_shared.hash_key_determinism import assert_hash_keys_are_deterministic
from py_ci_shared.check_then_act_outside_lock import assert_no_check_then_act_outside_lock
from py_ci_shared.fail_open_handlers import assert_no_new_fail_open_handlers
from py_ci_shared.identity_comparisons import assert_no_identity_comparisons
from py_ci_shared.import_cycles import assert_no_import_cycles
from py_ci_shared.module_cache_thread_safety import assert_thread_safe_module_caches
from py_ci_shared.lock_held_across_io import assert_no_lock_held_across_io
from py_ci_shared.module_reload_safety import assert_no_reloads_in_code
from py_ci_shared.naive_utcnow import assert_no_naive_utcnow
from py_ci_shared.no_source_text_tests import assert_no_source_text_tests
from py_ci_shared.no_xfail_to_defer import assert_no_xfail_to_defer
from py_ci_shared.numba_seed_range import assert_numba_seeds_fit_int64
from py_ci_shared.sentinel_or_fallback import assert_no_sentinel_or_fallback
from py_ci_shared.shared_mutable_placeholder import assert_no_shared_mutable_placeholders
from py_ci_shared.stale_source_citations import assert_no_stale_source_citations
from py_ci_shared.swallowed_exceptions import assert_no_swallowed_exceptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "pyutilz"
FAIL_OPEN_BASELINE = Path(__file__).resolve().parent / "_fail_open_baseline.json"


def test_numba_seeds_fit_the_seed_argument():
    """An entropy seed handed to numba must fit its int64 argument; a restore that overflows is swallowed and does nothing."""
    assert_numba_seeds_fit_int64(SRC, min_files=200)


def test_no_sentinel_setting_read_through_or():
    """`max_tokens or default` turns a meaningful 0 into the default; read the declared settings with an explicit None check."""
    assert_no_sentinel_or_fallback(SRC, min_files=200)


def test_source_citations_point_at_real_lines():
    """A `file.py:NNN` citation past the end of the file sends the reader nowhere; cite the symbol instead."""
    assert_no_stale_source_citations(SRC, min_files=200)


def test_module_caches_are_locked():
    """A module-level cache that inserts and evicts in one function needs a lock when scans run on a thread pool."""
    assert_thread_safe_module_caches(SRC, min_files=200)


def test_no_swallowed_io_or_broad_failures():
    """An `except OSError: continue` in a corpus scanner drops a file from the scan with no trace; say so, or mark it swallow-ok."""
    assert_no_swallowed_exceptions(SRC, min_files=200)


def test_no_xfail_that_defers_a_fix():
    """An xfail must be strict and name an external limit or a tracked issue, never park an in-repo bug."""
    assert_no_xfail_to_defer(REPO_ROOT / "tests", repo_root=REPO_ROOT, min_files=250)


def test_no_import_cycle():
    """No strongly connected import component, and no import that breaks when a cycle member is imported first."""
    assert_no_import_cycles(SRC, package="pyutilz", min_files=200)


def test_production_code_never_reloads_a_module():
    """Outside tests no fixture restores a reloaded module, so production code may not reload or drop one at all.

    The test-side half stays in test_no_module_reload.py: its per-site reviewed allowlist and its sys.modules-write check
    have no counterpart in the shared gate's test scan.
    """
    assert_no_reloads_in_code([SRC], REPO_ROOT, min_files=200)


def test_no_identity_comparison_of_string_constants():
    """`x is SOME_STRING` holds only while CPython happens to intern both sides; compare strings by value."""
    assert_no_identity_comparisons(sorted(SRC.rglob("*.py")), root=REPO_ROOT, min_files=200)


def test_every_counterpart_gate_named_by_code_audit_exists():
    """SHARED_GATE_COUNTERPARTS in pyutilz's code_audit registry points at gates by module name; each must import."""
    import importlib

    from pyutilz.dev.code_audit.registry import SHARED_GATE_COUNTERPARTS

    missing = []
    for gate in sorted(set(SHARED_GATE_COUNTERPARTS.values())):
        try:
            importlib.import_module(gate)
        except ImportError as exc:
            missing.append(f"{gate}: {exc}")
    assert missing == [], "counterpart gates that do not import: " + "; ".join(missing)


# Adopted 2026-09-26 (audit GEN-15): the four gates below cover bug classes that audit found by hand (a cache sidecar
# rewritten in place, a temp file shared by concurrent writers, a content hash over a summary). All at zero findings.


def test_cache_and_baseline_files_are_written_atomically():
    """A cache or baseline rewritten in place can be read truncated by another process; write a sibling and os.replace it."""
    assert_atomic_write_staging(SRC, min_files=200)


def test_json_fed_to_a_hash_sorts_its_keys():
    """`json.dumps(d)` follows insertion order, so equal dicts built in another order hash differently and the cache misses."""
    assert_hash_keys_are_deterministic(SRC, min_files=200)


def test_no_array_copied_just_to_be_hashed():
    """`h.update(a.tobytes())` allocates a full copy per cache lookup; hash the contiguous buffer instead."""
    assert_no_hash_fed_by_array_copy([SRC], min_files=200)


def test_no_naive_utcnow():
    """`datetime.utcnow()` is deprecated and naive; use `datetime.now(timezone.utc)`."""
    assert_no_naive_utcnow(SRC, min_files=200)


# Adopted with py-ci-shared 1.22.12 (the concurrency, placeholder, source-text and fail-open rules). Zero findings is the bar,
# except where the module docstring of the test says otherwise.


def _py_files(root: Path) -> list[Path]:
    """Every Python file under *root*, sorted so a finding's order is stable across runs."""
    return sorted(root.rglob("*.py"))


def test_no_lock_held_across_blocking_io():
    """A connect, HTTP call, subprocess, sleep or file write made while a lock is held makes every other thread wait for it."""
    assert_no_lock_held_across_io(_py_files(SRC), min_files=200)


def test_check_and_write_on_a_shared_dict_share_one_lock():
    """A membership test or get, then a write to the same shared dict, must sit under one lock block: another thread can insert between them."""
    assert_no_check_then_act_outside_lock(_py_files(SRC), min_files=200)


def test_no_mutable_placeholder_returned_by_reference():
    """A list, dict or object built once in an enclosing scope and returned by name hands every caller the same object."""
    assert_no_shared_mutable_placeholders(_py_files(SRC) + _py_files(REPO_ROOT / "tests"), min_files=200)


def test_no_test_asserts_on_sql_text():
    """A test that matches the spelling of a SQL statement proves nothing about the rows it returns; drive a fake or a real result instead."""
    assert_no_source_text_tests(_py_files(REPO_ROOT / "tests"), min_files=250)


def test_no_silent_default_return_in_except_handlers():
    """An except handler that returns a default and logs nothing hides the failure from the caller; log the exception type at the handler."""
    assert_no_new_fail_open_handlers(files=_py_files(SRC), repo_root=REPO_ROOT, baseline_path=FAIL_OPEN_BASELINE, min_files=200)
