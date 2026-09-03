"""Scanner tests for claimed_invariants, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_unenforced_docstring_invariants,
)

from ._helpers import _write

# ---- unenforced_docstring_invariant --------------------------------------


def test_unenforced_docstring_invariant_flags_an_unnamed_claim(tmp_path: Path):
    src = tmp_path / "src"
    tests_dir = tmp_path / "t"
    src.mkdir()
    tests_dir.mkdir()
    _write(src, "mod.py", '''
def decompose(x):
    """Never decompose a posterior produced by a different model."""
    return x
''')
    _write(tests_dir, "test_other.py", """
def test_something():
    assert True
""")
    findings = scan_unenforced_docstring_invariants(src, test_roots=(tests_dir,))
    assert len(findings) == 1 and "decompose" in findings[0].detail, findings


def test_unenforced_docstring_invariant_accepts_a_named_symbol(tmp_path: Path):
    src = tmp_path / "src"
    tests_dir = tmp_path / "t"
    src.mkdir()
    tests_dir.mkdir()
    _write(src, "mod.py", '''
def decompose(x):
    """Never decompose a posterior produced by a different model."""
    return x
''')
    _write(tests_dir, "test_mod.py", """
from mod import decompose


def test_decompose_refuses_a_foreign_model():
    assert decompose(1) == 1
""")
    assert scan_unenforced_docstring_invariants(src, test_roots=(tests_dir,)) == []


def test_unenforced_docstring_invariant_accepts_a_private_helper_via_its_public_caller(tmp_path: Path):
    """The common real shape: a PRIVATE helper's invariant is exercised indirectly, through tests
    that call the public function it lives inside rather than the private symbol directly - the
    normal way a private helper gets any test coverage at all. One hop only."""
    src = tmp_path / "src"
    tests_dir = tmp_path / "t"
    src.mkdir()
    tests_dir.mkdir()
    _write(src, "mod.py", '''
def _match_index(name):
    """EXACT name match only - never fuzzy/substring, an ambiguous name is refused rather than guessed."""
    return name


def build(name):
    return _match_index(name)
''')
    _write(tests_dir, "test_mod.py", """
from mod import build


def test_build_refuses_an_ambiguous_name():
    assert build("x") == "x"
""")
    assert scan_unenforced_docstring_invariants(src, test_roots=(tests_dir,)) == []


def test_unenforced_docstring_invariant_does_not_chase_two_hops(tmp_path: Path):
    """The deliberate stopping point: a private helper's caller's OWN caller being tested is not
    enough - chasing two hops starts matching chains a reader would not recognise as "this test
    covers that claim" on inspection, the same false-confidence failure this check exists to catch."""
    src = tmp_path / "src"
    tests_dir = tmp_path / "t"
    src.mkdir()
    tests_dir.mkdir()
    _write(src, "mod.py", '''
def _match_index(name):
    """EXACT name match only - never fuzzy/substring."""
    return name


def _build_one(name):
    return _match_index(name)


def build_all(names):
    return [_build_one(n) for n in names]
''')
    _write(tests_dir, "test_mod.py", """
from mod import build_all


def test_build_all_refuses_an_ambiguous_name():
    assert build_all(["x"]) == ["x"]
""")
    findings = scan_unenforced_docstring_invariants(src, test_roots=(tests_dir,))
    assert len(findings) == 1 and "_match_index" in findings[0].detail, findings
