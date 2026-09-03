"""Scanner tests for redundant_test_fit, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

from pyutilz.dev.code_audit import (
    scan_redundant_test_fit_calls,
)

from ._helpers import _scanner_function, _write

# ---- subpackage facade sensor ------------------------------------------


# ---- redundant_test_fit_call ---------------------------------------------


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_identical_fit_call_across_two_tests_flags(tmp_path: Path):
    """The exact confirmed-real-bug shape (mlframe MRMR biz_value suite): two sibling
    test functions each independently call the SAME deterministic helper with the SAME
    literal seed to check a different assertion on the identical fit result."""
    _write(tmp_path, "test_bad.py", """
def _build_data(seed):
    return seed

def _fit_model(X, seed):
    return X + seed

def test_a():
    X = _build_data(seed=101)
    sel = _fit_model(X, seed=101)
    assert sel

def test_b():
    X = _build_data(seed=101)
    sel = _fit_model(X, seed=101)
    assert sel
""")
    findings = scan_redundant_test_fit_calls(tmp_path)
    checks = {f.check for f in findings}
    assert "redundant_test_fit_call" in checks
    assert findings and all(f.severity == "Low" for f in findings)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_loop_variable_argument_is_not_a_repeated_call(tmp_path: Path):
    """Two functions each looping over their own data unparse to the same signature - `_resolves(x)` -
    while sharing no call at all. The check is for a deterministic repeat of ONE call, and a loop variable
    takes a different value every iteration."""
    _write(tmp_path, "test_loop.py", """
def _resolves(x):
    return bool(x)

def test_a():
    assert all(_resolves(x) for x in [1, 2, 3])

def test_b():
    bad = [x for x in [4, 5] if not _resolves(x)]
    assert not bad
""")
    assert [f for f in scan_redundant_test_fit_calls(tmp_path) if f.check == "redundant_test_fit_call"] == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_module_level_constant_argument_is_still_a_repeated_call(tmp_path: Path):
    """The exemption must key on ITERATION binding, not on "the argument is a name": a constant passed by
    name really is the same value in both callers, which is the genuine duplicate this check exists for."""
    _write(tmp_path, "test_const.py", """
SEED = 101

def _fit(seed):
    return seed

def test_a():
    assert _fit(SEED)

def test_b():
    assert _fit(SEED) == 101
""")
    assert "redundant_test_fit_call" in {f.check for f in scan_redundant_test_fit_calls(tmp_path)}


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_literal_data_factory_is_not_flagged(tmp_path: Path):
    """A helper that fills in a dict literal is the opposite of the expensive fit this scanner hunts: it
    costs microseconds, and its result is a FRESH MUTABLE object each caller then edits. Acting on the
    finding here - caching it, or sharing one fixture - would hand every test the same dict and let one
    test's mutation reach another, so a flag on this shape recommends a bug."""
    _write(tmp_path, "test_factory.py", """
def _item(**over):
    base = {"name": "x", "hits": 13, "denominator": 142}
    base.update(over)
    return base

def test_a():
    assert _item()["hits"] == 13

def test_b():
    assert _item()["denominator"] == 142
""")
    assert [f for f in scan_redundant_test_fit_calls(tmp_path) if f.check == "redundant_test_fit_call"] == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_helper_with_no_calls_is_still_flagged(tmp_path: Path):
    """The exemption must key on BUILDING a literal, not merely on containing no expensive-looking call:
    `def _build_data(seed): return seed` is what an expensive builder is reduced to in a scanner test."""
    _write(tmp_path, "test_stub.py", """
def _build_data(seed):
    return seed

def test_a():
    assert _build_data(seed=101) == 101

def test_b():
    assert _build_data(seed=101) is not None
""")
    assert "redundant_test_fit_call" in {f.check for f in scan_redundant_test_fit_calls(tmp_path)}


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_different_seeds_not_flagged(tmp_path: Path):
    """Different literal args -> genuinely different computations, not a duplicate."""
    _write(tmp_path, "test_ok.py", """
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=202)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_same_test_not_flagged(tmp_path: Path):
    """The SAME call appearing twice within one test function (e.g. a sanity re-check)
    is not a cross-test duplication -- only 2+ DIFFERENT test functions count."""
    _write(tmp_path, "test_ok.py", """
def _build_data(seed):
    return seed

def test_a():
    X1 = _build_data(seed=101)
    X2 = _build_data(seed=101)
    assert X1 == X2
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_already_cached_not_flagged(tmp_path: Path):
    """A helper already decorated with @cache/@lru_cache has already been fixed."""
    _write(tmp_path, "test_ok.py", """
from functools import cache

@cache
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_redundant_call_non_test_file_not_scanned(tmp_path: Path):
    """This scanner only applies to test_*.py / *_test.py files."""
    _write(tmp_path, "helpers.py", """
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_redundant_call_public_function_not_flagged(tmp_path: Path):
    """Only underscore-prefixed local helpers are tracked -- repeated identical calls to a
    public/third-party-style function (no leading underscore) are a normal, cheap pattern."""
    _write(tmp_path, "test_ok.py", """
def build_data(seed):
    return seed

def test_a():
    X = build_data(seed=101)
    assert X

def test_b():
    X = build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_facade_reexports_are_same_objects():
    """After the >1000-LOC split into a subpackage, the ``code_audit``
    facade must re-export every public symbol as the SAME object the
    cohesive submodule defines. Guards against a future submodule shuffle
    silently changing the public import surface."""
    import pyutilz.dev.code_audit as facade
    from pyutilz.dev.code_audit._base import Finding as _Finding
    from pyutilz.dev.code_audit.mutable_defaults import scan_mutable_defaults as _smd
    from pyutilz.dev.code_audit.closures import scan_late_binding_closures as _slbc
    from pyutilz.dev.code_audit.default_via_or import scan_default_via_or_trap as _sdvot
    from pyutilz.dev.code_audit.broad_except import scan_broad_except_swallows as _sbes
    from pyutilz.dev.code_audit.nan_equality import scan_nan_equality as _sne
    from pyutilz.dev.code_audit.mutation_during_iteration import scan_mutation_during_iteration as _smdi
    from pyutilz.dev.code_audit.sql_lint import scan_sql_limit_without_order_by as _sslwob, scan_sql_offset_pagination as _ssop
    from pyutilz.dev.code_audit.dead_cli_flags import scan_dead_cli_flags as _sdcf
    from pyutilz.dev.code_audit.silent_escalation import scan_log_only_except as _sloe, DEFAULT_ESCALATION_ATTRS as _DEA
    from pyutilz.dev.code_audit.sql_migrations import scan_sql_migration_idempotency as _ssmi
    from pyutilz.dev.code_audit.duplicate_conditions import scan_duplicate_conditions as _sdc
    from pyutilz.dev.code_audit.missed_await import scan_missed_await as _sma
    from pyutilz.dev.code_audit.redundant_test_fit import scan_redundant_test_fit_calls as _srtfc
    from pyutilz.dev.code_audit.registry import run_all as _ra, get_scanners as _get_scanners
    from pyutilz.dev.code_audit.cli import main as _main

    assert facade.Finding is _Finding
    assert facade.scan_mutable_defaults is _smd
    assert facade.scan_late_binding_closures is _slbc
    assert facade.scan_default_via_or_trap is _sdvot
    assert facade.scan_broad_except_swallows is _sbes
    assert facade.scan_nan_equality is _sne
    assert facade.scan_mutation_during_iteration is _smdi
    assert facade.scan_sql_limit_without_order_by is _sslwob
    assert facade.scan_sql_offset_pagination is _ssop
    assert facade.scan_dead_cli_flags is _sdcf
    assert facade.scan_log_only_except is _sloe
    assert facade.DEFAULT_ESCALATION_ATTRS is _DEA
    assert facade.scan_sql_migration_idempotency is _ssmi
    assert facade.scan_duplicate_conditions is _sdc
    assert facade.scan_missed_await is _sma
    assert facade.scan_redundant_test_fit_calls is _srtfc
    assert facade.run_all is _ra
    assert facade.get_scanners is _get_scanners
    # The mutable registry itself is NOT part of the facade -- only the read-only accessor is.
    assert not hasattr(facade, "SCANNERS")
    assert facade.main is _main
    # Every scanner in the registry is the facade-level attribute of the same name -- see
    # test_registry_and_facade_are_in_bijection for the full both-ways invariant.
    for fn in facade.get_scanners().values():
        assert callable(fn)
        assert getattr(facade, _scanner_function(fn).__name__, None) is _scanner_function(fn)


def test_cli_json_output(tmp_path: Path, capsys):
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    from pyutilz.dev.code_audit import main as cli_main
    cli_main([str(tmp_path), "--format", "json"])
    import json as _json
    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert isinstance(payload, list)
    assert payload and payload[0]["check"] == "mutable_default"


# ---- F63: redundant test fit, on the 3.8 floor --------------------------------------


def test_redundant_test_fit_calls_works_without_ast_unparse(tmp_path: Path, monkeypatch):
    """python 3.8 has no ast.unparse; the scanner must not degrade to a silent no-op there."""
    import ast as _ast

    _write(tmp_path, "test_a.py", """
def _build_data(n, seed=0):
    return compute(n, seed)


def test_a():
    d = _build_data(100, seed=0)
    assert d


def test_b():
    d = _build_data(100, seed=0)
    assert d
""")
    monkeypatch.delattr(_ast, "unparse", raising=False)
    assert len(scan_redundant_test_fit_calls(tmp_path)) == 1


def test_redundant_test_fit_calls_ignores_different_arguments(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
def _build_data(n):
    return compute(n)


def test_a():
    assert _build_data(1)


def test_b():
    assert _build_data(2)
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []
