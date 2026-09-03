"""Scanner tests for vacuous_loop_assertion, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.vacuous_loop_assertion import scan_vacuous_loop_assertion

from ._helpers import _write

# ---- vacuous_loop_assertion ----------------------------------------------


def test_vacuous_loop_assertion_flags_the_canonical_case(tmp_path: Path):
    """Every assertion inside a loop over a call's result, with nothing pinning the count."""
    _write(
        tmp_path,
        "test_rows.py",
        """
def test_rows(db):
    for row in db.fetch_all():
        assert row.uid
""",
    )
    findings = scan_vacuous_loop_assertion(tmp_path)
    assert len(findings) == 1, findings
    assert "test_rows" in findings[0].detail


def test_vacuous_loop_assertion_accepts_a_count_first(tmp_path: Path):
    """`assert lst` before the loop is the whole fix, and it is an assertion outside the loop.

    The rule needs no separate guard check for this, and an earlier draft that had one could never
    fire: the two conditions overlap completely, so removing either left the other holding.
    """
    _write(
        tmp_path,
        "test_rows.py",
        """
def test_rows(db, lst):
    assert lst
    for chunk in chunks(lst, 10):
        assert chunk
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []


def test_vacuous_loop_assertion_accepts_an_assertion_outside(tmp_path: Path):
    """An assertion the loop does not own holds the test up whatever the loop does."""
    _write(
        tmp_path,
        "test_rows.py",
        """
def test_rows(db):
    assert db.is_connected()
    for row in db.fetch_all():
        assert row.uid
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []


def test_vacuous_loop_assertion_ignores_an_imported_registry(tmp_path: Path):
    """A registry defined anywhere in the tree cannot be empty, and iterating it is not a defect.

    Resolved only within the test file, `_ALIASES.items()` and `_CC_REGISTRY.items()` read as
    collections that might be empty; they were the bulk of this rule's hits before the name
    resolution went package-wide.
    """
    _write(tmp_path, "registry.py", "_ALIASES = {'a': 'alpha', 'b': 'beta'}")
    _write(
        tmp_path,
        "test_aliases.py",
        """
from registry import _ALIASES


def test_aliases():
    for alias, canonical in _ALIASES.items():
        assert canonical
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []


def test_vacuous_loop_assertion_ignores_an_index_loop(tmp_path: Path):
    """`for i in range(len(df))` is an index loop; what could be empty is decided elsewhere."""
    _write(
        tmp_path,
        "test_frame.py",
        """
def test_frame(df):
    for i in range(len(df)):
        assert df.iloc[i] is not None
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []


def test_vacuous_loop_assertion_ignores_a_literal(tmp_path: Path):
    """A literal with something in it cannot iterate zero times."""
    _write(
        tmp_path,
        "test_modes.py",
        """
def test_modes(run):
    for mode in ["fast", "slow"]:
        assert run(mode)
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []


def test_vacuous_loop_assertion_only_reads_test_functions(tmp_path: Path):
    """A helper that loops and asserts is not a test that stopped testing.

    Replaces a version that put the same code in a non-test FILE: the filename filter it claimed
    to exercise was dead, because the function-name check already excluded it.
    """
    _write(
        tmp_path,
        "test_rows.py",
        """
def check_rows(db):
    for row in db.fetch_all():
        assert row.uid
""",
    )
    assert scan_vacuous_loop_assertion(tmp_path) == []
