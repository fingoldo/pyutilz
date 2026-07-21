"""Unit tests for pyutilz.performance.kernel_tuning.code_versioning.compute_code_version.

The hashed functions are defined at MODULE level so inspect.getsource can
retrieve their source (functions defined in a ``python -c`` string or the REPL
have no retrievable source and fall back to the identity hash)."""
import pytest

from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version as cv


def _f1(x):
    # a comment that must not affect the hash
    return x + 1


def _f1_renamed_reformatted(x):
    "a docstring that must not affect the hash"
    return x + 1  # reformatted + renamed, SAME logic


def _f2(x):
    return x + 2  # different logic


# numba kernels (module level; .py_func source must be retrievable). Defined in
# a guard so the non-numba tests still run when numba is unavailable.
try:
    import numba  # noqa: F401

    @numba.njit(cache=False)
    def _g(x):
        s = 0.0
        for i in range(x.shape[0]):
            s += x[i]
        return s

    @numba.njit(cache=False)
    def _g_renamed(x):
        # same body as _g, different name + comment
        s = 0.0
        for i in range(x.shape[0]):
            s += x[i]
        return s

    @numba.njit(cache=False)
    def _g_diff(x):
        return x.sum()

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False


def test_returns_hex_sha256():
    h = cv(_f1)
    assert isinstance(h, str) and len(h) == 64
    int(h, 16)  # parses as hex


def test_logic_only_rename_format_docstring_insensitive():
    # A rename + reformat + docstring is NOT a logic change -> same version.
    assert cv(_f1) == cv(_f1_renamed_reformatted)


def test_logic_change_detected():
    assert cv(_f1) != cv(_f2)


def test_salt_changes_hash():
    assert cv(_f1) != cv(_f1, salt=1)
    assert cv(_f1, salt=2) != cv(_f1, salt=3)


def test_extra_fns_change_hash():
    assert cv(_f1) != cv(_f1, extra_fns=(_f2,))


def test_order_independent():
    assert cv(_f1, _f2) == cv(_f2, _f1)


def test_deterministic():
    assert cv(_f1, _f2, salt=7) == cv(_f1, _f2, salt=7)


def test_graceful_fallback_when_source_unavailable():
    # A builtin has no Python source -> identity fallback; must not raise and
    # must still return a valid 64-char sha256.
    h = cv(len)
    assert isinstance(h, str) and len(h) == 64


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba unavailable")
def test_numba_pyfunc_unwrap_and_logic_sensitivity():
    # Source is retrieved via Dispatcher.py_func -> rename-insensitive equality,
    # logic-sensitive inequality, and distinct from a pure-python function.
    assert cv(_g) == cv(_g_renamed)
    assert cv(_g) != cv(_g_diff)
    assert cv(_g) != cv(_f1)


def test_inplace_code_swap_invalidates_cache():
    """Regression (2026-07-21 audit round 2, HIGH): the old design put a single ``lru_cache``
    directly on ``compute_code_version``, keyed by the function-object args -- identity-based,
    so it was blind to an in-place ``fn.__code__`` swap (IPython's ``%autoreload 2`` mechanism:
    same function object, different bytecode). ``compute_code_version`` is no longer
    ``lru_cache``d itself; the per-function source lookup is cached content-aware via
    ``_code_identity_key``/``_normalized_source_by_code`` instead."""
    before = cv(_f1)
    original_code = _f1.__code__
    try:
        _f1.__code__ = _f2.__code__  # simulate autoreload: same object, swapped-in logic
        after = cv(_f1)
        assert after != before, "hash must change once the function's actual bytecode changes"
    finally:
        _f1.__code__ = original_code  # restore -- _f1 is module-level, shared by other tests
    # And restoring the original code must make the hash match the original again --
    # proving this isn't a one-way "always recompute" degradation either.
    assert cv(_f1) == before


def test_repeated_calls_with_unchanged_code_hit_the_per_function_cache():
    """Perf sanity: the expensive per-function AST parse/unparse (~1.3ms) is still memoized
    when a function's code object hasn't changed -- only the outer identity-blind cache was
    removed, not caching altogether."""
    from pyutilz.performance.kernel_tuning.code_versioning import _normalized_source_by_code

    _normalized_source_by_code.cache_clear()
    cv(_f1)
    hits_before = _normalized_source_by_code.cache_info().hits
    cv(_f1)
    cv(_f1)
    hits_after = _normalized_source_by_code.cache_info().hits
    assert hits_after > hits_before
