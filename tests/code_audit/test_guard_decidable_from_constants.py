"""Scanner tests for guard_decidable_from_constants, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.guard_decidable_from_constants import scan_guard_decidable_from_constants

from ._helpers import _write

# ---- guard_decidable_from_constants --------------------------------------


def test_guard_decidable_from_constants_flags_a_dead_branch(tmp_path: Path):
    """The shape it exists for: a private literal deciding a guard, written by nothing.

    Real instance in mlframe: `_KNOCKOFFS_STRICT_LAM_MIN = False` whose comment promises it is
    "set via globals().setdefault from the call site" -- and no such write exists anywhere in
    that repository, so the `raise ValueError` it guards has never run.
    """
    _write(
        tmp_path,
        "out.py",
        """
_STRICT = False

def check(value):
    if _STRICT:
        raise ValueError(value)
    return value
""",
    )
    findings = scan_guard_decidable_from_constants(tmp_path)
    assert len(findings) == 1, findings
    assert "_STRICT" in findings[0].detail


def test_guard_decidable_from_constants_ignores_a_public_knob(tmp_path: Path):
    """A public module-level name is set by importers -- `browser.undetectable = True` before
    calling `start_selenium()` is how this package's own selenium module is driven. That one
    pattern supplied eight of this rule's first eight hits."""
    _write(
        tmp_path,
        "out.py",
        """
undetectable = False

def start():
    if undetectable:
        return "stealth"
    return "plain"
""",
    )
    assert scan_guard_decidable_from_constants(tmp_path) == []


def test_guard_decidable_from_constants_sees_a_write_from_a_sibling_module(tmp_path: Path):
    """Package-wide, not per-module: another file rebinding the name by attribute is invisible
    to a walk of the defining module, and four of this rule's first hits were exactly that."""
    _write(
        tmp_path,
        "kernels.py",
        """
_THREADS_OVERRIDE = None

def threads():
    if _THREADS_OVERRIDE is not None:
        return _THREADS_OVERRIDE
    return 128
""",
    )
    _write(
        tmp_path,
        "sweep.py",
        """
import kernels

def tune(n):
    kernels._THREADS_OVERRIDE = n
""",
    )
    assert scan_guard_decidable_from_constants(tmp_path) == []


def test_guard_decidable_from_constants_ignores_an_optional_import_probe(tmp_path: Path):
    """`spacy = None` reassigned inside a `try:` is the canonical optional-dependency probe, and
    the assignment that matters is nested rather than in the module statement list."""
    _write(
        tmp_path,
        "out.py",
        """
_spacy = None
try:
    import spacy as _real
    _spacy = _real
except Exception:
    pass

def tokenize(text):
    if _spacy is None:
        return text.split()
    return _spacy(text)
""",
    )
    assert scan_guard_decidable_from_constants(tmp_path) == []


def test_guard_decidable_from_constants_ignores_a_non_literal(tmp_path: Path):
    """A name computed at import time is not a literal, so nothing about it is decided."""
    _write(
        tmp_path,
        "out.py",
        """
import os

_STRICT = os.environ.get("STRICT") == "1"

def check(value):
    if _STRICT:
        raise ValueError(value)
    return value
""",
    )
    assert scan_guard_decidable_from_constants(tmp_path) == []
