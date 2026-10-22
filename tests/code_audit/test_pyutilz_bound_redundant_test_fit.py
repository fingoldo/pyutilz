"""The tests split out of test_redundant_test_fit.py that touch pyutilz-only code (field_text_agreement, the CLI, the pyutilz facade); the rest moved to py-ci-shared."""

from __future__ import annotations


from pathlib import Path


from ._helpers import _scanner_function, _write


def test_facade_reexports_are_same_objects():
    """After the >1000-LOC split into a subpackage, the ``code_audit``
    facade must re-export every public symbol as the SAME object the
    cohesive submodule defines. Guards against a future submodule shuffle
    silently changing the public import surface."""
    import py_ci_shared.code_audit as facade
    from py_ci_shared.code_audit._base import Finding as _Finding
    from py_ci_shared.code_audit.mutable_defaults import scan_mutable_defaults as _smd
    from py_ci_shared.code_audit.closures import scan_late_binding_closures as _slbc
    from py_ci_shared.code_audit.default_via_or import scan_default_via_or_trap as _sdvot
    from py_ci_shared.code_audit.broad_except import scan_broad_except_swallows as _sbes
    from py_ci_shared.code_audit.nan_equality import scan_nan_equality as _sne
    from py_ci_shared.code_audit.mutation_during_iteration import scan_mutation_during_iteration as _smdi
    from py_ci_shared.code_audit.sql_lint import scan_sql_limit_without_order_by as _sslwob, scan_sql_offset_pagination as _ssop
    from py_ci_shared.code_audit.dead_cli_flags import scan_dead_cli_flags as _sdcf
    from py_ci_shared.code_audit.silent_escalation import scan_log_only_except as _sloe, DEFAULT_ESCALATION_ATTRS as _DEA
    from py_ci_shared.code_audit.sql_migrations import scan_sql_migration_idempotency as _ssmi
    from py_ci_shared.code_audit.duplicate_conditions import scan_duplicate_conditions as _sdc
    from py_ci_shared.code_audit.missed_await import scan_missed_await as _sma
    from py_ci_shared.code_audit.redundant_test_fit import scan_redundant_test_fit_calls as _srtfc
    from py_ci_shared.code_audit.registry import run_all as _ra, get_scanners as _get_scanners

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
    # Every scanner in the registry is the facade-level attribute of the same name -- see
    # test_registry_and_facade_are_in_bijection for the full both-ways invariant.
    scanners = facade.get_scanners()
    assert scanners, "the registry exposes no scanners"
    for fn in scanners.values():
        assert callable(fn)
        assert getattr(facade, _scanner_function(fn).__name__, None) is _scanner_function(fn)


def test_cli_json_output(tmp_path: Path, capsys):
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    from py_ci_shared.code_audit.cli import main as cli_main
    cli_main([str(tmp_path), "--format", "json"])
    from pyutilz.core import jsonlib as _jsonlib
    out = capsys.readouterr().out
    payload = _jsonlib.loads(out)
    assert isinstance(payload, list)
    assert payload and payload[0]["check"] == "mutable_default"
