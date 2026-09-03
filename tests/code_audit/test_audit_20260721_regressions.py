"""Scanner tests for audit_20260721_regressions, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path
import pytest

from pyutilz.dev.code_audit import (
    Finding,
    scan_dead_cli_flags,
    scan_duplicate_conditions,
    scan_late_binding_closures,
    scan_missed_await,
    scan_mutable_defaults,
    scan_mutation_during_iteration,
    scan_nan_equality,
    scan_sql_migration_idempotency,
)

from ._helpers import _write

# --- 2026-07-21 audit regression tests ------------------------------------


def test_cli_min_severity_does_not_weaken_exit_code(tmp_path: Path):
    """Regression: --min-severity previously filtered `findings` BEFORE the exit-code check,
    so a real P1 finding silently exited 0 once filtered out of the display."""
    from pyutilz.dev.code_audit import main as cli_main

    _write(tmp_path, "bad.py", """
async def process(item):
    await item.save()

def caller(item):
    process(item)
""")
    assert cli_main([str(tmp_path), "--min-severity", "Low"]) == 1
    assert cli_main([str(tmp_path), "--min-severity", "P0"]) == 1


def test_mutable_default_not_flagged_when_only_shadowing_nested_func_mutates(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def outer(x=[]):
    def inner(x):
        x.append(1)
        return x
    return inner([1, 2, 3])
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"  # not P0: outer's own x is never mutated


def test_late_binding_closure_flags_list_comprehension(tmp_path: Path):
    _write(tmp_path, "bad.py", """
handlers = [lambda: x for x in range(3)]
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_late_binding_closure_flags_dict_comprehension(tmp_path: Path):
    _write(tmp_path, "bad.py", """
handlers = {i: (lambda: i) for i in range(3)}
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1


def test_missed_await_not_flagged_when_shadowed_by_nested_def(tmp_path: Path):
    _write(tmp_path, "ok.py", """
async def process(item):
    await item.save()

def sync_wrapper(item):
    def process(x):
        x.touch()
    process(item)
""")
    findings = scan_missed_await(tmp_path)
    assert findings == []


def test_dead_cli_flag_not_flagged_when_read_via_literal_getattr(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
if getattr(args, "resume"):
    print("resuming")
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert findings == []


def test_sql_migration_recognizes_custom_dollar_quote_tag(tmp_path: Path):
    (tmp_path / "migration.sql").write_text(
        """
DO $body$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'pk_users'
    ) THEN
        ALTER TABLE users ADD PRIMARY KEY (id);
    END IF;
END $body$;
""",
        encoding="utf-8",
    )
    findings = scan_sql_migration_idempotency(tmp_path)
    assert findings == []


def test_finding_as_md_row_escapes_pipe_in_detail():
    f = Finding(check="x", severity="Low", file="a.py", line=1, snippet="s", detail="an `X | None` parameter")
    row = f.as_md_row()
    assert "X \\| None" in row
    # Table structure preserved: exactly 4 unescaped pipes delimit the 5 cells (plus outer edges).
    assert row.count("|") - row.count("\\|") == 6


def test_registry_register_scanner_rejects_collision():
    from pyutilz.dev.code_audit.registry import register_scanner, get_scanners

    def _dummy(root, exclude_dirs=frozenset()):
        return []

    with pytest.raises(ValueError):
        register_scanner("mutable_default", _dummy)
    assert get_scanners()["mutable_default"] is not _dummy

    register_scanner("mutable_default", _dummy, allow_override=True)
    try:
        assert get_scanners()["mutable_default"] is _dummy
    finally:
        register_scanner("mutable_default", scan_mutable_defaults, allow_override=True)


def test_duplicate_conditions_not_flagged_for_impure_bare_function_retry(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    success = attempt() or attempt()
    return success
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert findings == []


def test_nan_equality_ignores_unrelated_dot_nan_attribute(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(result, expected):
    if result.nan == expected.nan:
        return True
    return False
""")
    findings = scan_nan_equality(tmp_path)
    assert findings == []


def test_nan_equality_still_flags_np_nan(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np
def f(x):
    if x == np.nan:
        return True
    return False
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1


def test_mutation_during_iteration_list_message_is_backend_agnostic(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(mylist):
    for x in mylist:
        mylist[0] = x * 2
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert "RuntimeError on dict/set" not in findings[0].detail
