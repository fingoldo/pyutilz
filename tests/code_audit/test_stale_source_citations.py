"""Scanner tests for stale_source_citation: the two exact tiers, resolution order, and what stays silent."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_stale_source_citations

from ._helpers import _write


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg" / "db").mkdir(parents=True)
    (tmp_path / "pkg" / "db" / "helpers.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    return tmp_path


def test_a_citation_of_a_file_that_is_gone_is_reported(tmp_path: Path):
    """The split-into-a-package case: `db/validation.py` became `db/validation/`, the comment kept the old path."""
    _write(_tree(tmp_path), "pkg/user.py", "x = 1  # mirrors db/validation.py:70\n")
    findings = scan_stale_source_citations(tmp_path)
    assert [(f.check, f.line) for f in findings] == [("stale_source_citation", 1)]
    assert "`db/validation.py`" in findings[0].detail


def test_a_line_past_the_end_is_reported_and_a_range_uses_its_end(tmp_path: Path):
    _write(_tree(tmp_path), "pkg/user.py", "x = 1  # see db/helpers.py:9\ny = 2  # see db/helpers.py:2-7\n")
    findings = scan_stale_source_citations(tmp_path)
    assert [f.line for f in findings] == [1, 2]
    assert "has 3 lines" in findings[0].detail


def test_a_citation_that_still_resolves_is_clean(tmp_path: Path):
    """Relative to the root, to the citing folder, and by a unique suffix: all three resolve."""
    _write(_tree(tmp_path), "pkg/user.py", '''
"""See pkg/db/helpers.py:3 and helpers.py:2."""
x = 1  # db/helpers.py:1
''')
    assert scan_stale_source_citations(tmp_path) == []


def test_code_strings_urls_and_other_repositories_are_not_citations(tmp_path: Path):
    _write(_tree(tmp_path), "pkg/user.py", '''
PATH = "db/gone.py:10"  # a string in code, not a comment
# https://example.com/raw/gone.py:10 is a URL
# py-ci-shared/src/py_ci_shared/gone.py:10 lives in another repository
''')
    assert scan_stale_source_citations(tmp_path) == []


def test_a_docstring_is_read_and_a_string_argument_is_not(tmp_path: Path):
    """A test fixture writing a file that contains a citation is data; a docstring is prose about this code.

    A first sweep of the scanner over its own repository flagged its own test fixtures, all string arguments.
    """
    _write(_tree(tmp_path), "pkg/user.py", '''
def f():
    """Kept in step with db/gone.py:10."""

write("pkg/x.py", """
# see db/gone.py:10
""")
''')
    findings = scan_stale_source_citations(tmp_path)
    assert [f.line for f in findings] == [2]


def test_an_ambiguous_basename_is_not_this_scanners_call(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "utils.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b" / "utils.py").write_text("x = 1\n", encoding="utf-8")
    _write(tmp_path, "user.py", "# utils.py:500\n")
    assert scan_stale_source_citations(tmp_path) == []
