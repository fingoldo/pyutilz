"""Scanner tests for comment_names_missing_symbol, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.comment_names_missing_symbol import (
    scan_comment_cites_absolute_line,
    scan_comment_names_missing_symbol,
)

from ._helpers import _write

# ---- comment_names_missing_symbol ----------------------------------------
#
# Prose that points somewhere is trusted. One such comment WAS the accepted mitigation for an
# earlier SQL-injection finding and named a helper that had since been renamed.


def test_comment_names_missing_symbol_flags_a_rotted_private_pointer(tmp_path: Path):
    _write(
        tmp_path,
        "perm.py",
        """
# The SQL is built by `_perm_err_sql()`, which escapes every pattern.
def perm_err_text_like_sql(patterns):
    return " OR ".join(patterns)
""",
    )
    findings = scan_comment_names_missing_symbol(tmp_path)
    assert len(findings) == 1
    assert "_perm_err_sql" in findings[0].detail


def test_comment_names_missing_symbol_ignores_library_methods(tmp_path: Path):
    """Unrestricted, this rule gave 52 hits in one package with no rotted pointer among them:
    `close()`, `min()`, `utcnow()`, `is_nan()`, `to_plotly_json()`, `model_dump()`. A leading
    underscore is the only reliable "this must be local" signal."""
    _write(
        tmp_path,
        "frames.py",
        """
# Values are dropped with `dropna()` and checked with `is_nan()` before `to_numpy()`.
def clean(df):
    return df
""",
    )
    assert scan_comment_names_missing_symbol(tmp_path) == []


def test_comment_names_missing_symbol_resolves_across_the_tree(tmp_path: Path):
    """A comment may cite a private helper defined in another module."""
    _write(
        tmp_path,
        "helpers.py",
        """
def _capped(n):
    return min(n, 100)
""",
    )
    _write(
        tmp_path,
        "user.py",
        """
# Capped by `_capped()` before use.
def go(n):
    return n
""",
    )
    assert scan_comment_names_missing_symbol(tmp_path) == []


def test_comment_cites_absolute_line_is_opt_in():
    """225 hits in one package, most of them legitimate coverage annotations. It reports rather
    than gates, so it cannot reach a project's default run or its baseline."""
    from pyutilz.dev.code_audit import OPT_IN_ONLY, get_scanners

    assert "comment_cites_absolute_line" in get_scanners()
    assert "comment_cites_absolute_line" in OPT_IN_ONLY


def test_comment_cites_absolute_line_finds_a_citation(tmp_path: Path):
    _write(
        tmp_path,
        "mod.py",
        """
# The unlink happens at line 619, after the flush.
def go():
    pass
""",
    )
    findings = scan_comment_cites_absolute_line(tmp_path)
    assert len(findings) == 1 and "619" in findings[0].detail
