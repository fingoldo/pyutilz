"""Scanner tests for sibling_guard_missing, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.sibling_guard_missing import scan_sibling_guard_missing

from ._helpers import _write

# ---- sibling_guard_missing -----------------------------------------------


def test_sibling_guard_missing_flags_the_odd_one_out(tmp_path: Path):
    """Three siblings open `if self._closed: return`; the fourth touches the same handle and does not."""
    _write(
        tmp_path,
        "handle.py",
        """
class Handle:
    def read(self):
        if self._closed:
            return None
        return self._fd.read()

    def write(self, data):
        if self._closed:
            return None
        return self._fd.write(data)

    def flush(self):
        if self._closed:
            return None
        return self._fd.flush()

    def truncate(self):
        return self._fd.truncate()
""",
    )
    findings = scan_sibling_guard_missing(tmp_path)
    assert len(findings) == 1, findings
    assert "truncate" in findings[0].detail and "_fd" in findings[0].detail


def test_sibling_guard_missing_needs_three_siblings(tmp_path: Path):
    """Two is a coincidence; three is a convention."""
    _write(
        tmp_path,
        "handle.py",
        """
class Handle:
    def read(self):
        if self._closed:
            return None
        return self._fd.read()

    def write(self, data):
        if self._closed:
            return None
        return self._fd.write(data)

    def flush(self):
        return self._fd.flush()

    def truncate(self):
        return self._fd.truncate()
""",
    )
    assert scan_sibling_guard_missing(tmp_path) == []


def test_sibling_guard_missing_accepts_a_guard_spelled_differently(tmp_path: Path):
    """A check written another way, or later in the body, is still a check.

    Reporting it would be telling the author to write the guard they already wrote.
    """
    _write(
        tmp_path,
        "handle.py",
        """
class Handle:
    def read(self):
        if self._closed:
            return None
        return self._fd.read()

    def write(self, data):
        if self._closed:
            return None
        return self._fd.write(data)

    def flush(self):
        if self._closed:
            return None
        return self._fd.flush()

    def truncate(self):
        if not self._closed:
            return self._fd.truncate()
        return None
""",
    )
    assert scan_sibling_guard_missing(tmp_path) == []


def test_sibling_guard_missing_needs_a_shared_object_attribute(tmp_path: Path):
    """A sibling that touches none of the guarded state has no business being guarded."""
    _write(
        tmp_path,
        "handle.py",
        """
class Handle:
    def read(self):
        if self._closed:
            return None
        return self._fd.read()

    def write(self, data):
        if self._closed:
            return None
        return self._fd.write(data)

    def flush(self):
        if self._closed:
            return None
        return self._fd.flush()

    def describe(self):
        return self._name
""",
    )
    assert scan_sibling_guard_missing(tmp_path) == []


def test_sibling_guard_missing_ignores_names_that_are_not_object_state(tmp_path: Path):
    """The link must be the object's own state, never any shared name.

    Intersecting all names produced resources like `['os', 'path']` and
    `['Any', 'Dict', 'Optional', 'ValueError']` -- imported type names and builtins appear in every
    method of a class, so the intersection is never empty and the link it forms means nothing.
    """
    _write(
        tmp_path,
        "writer.py",
        """
import os


class Writer:
    def save(self, path):
        if self._in_memory:
            return None
        return os.replace(path, path)

    def load(self, path):
        if self._in_memory:
            return None
        return os.stat(path)

    def clear(self, path):
        if self._in_memory:
            return None
        return os.remove(path)

    def atomic_write(self, path):
        return os.replace(path, path)
""",
    )
    assert scan_sibling_guard_missing(tmp_path) == []
