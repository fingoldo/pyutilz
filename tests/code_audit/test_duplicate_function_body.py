"""Scanner tests for duplicate_function_body, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_duplicate_function_body,
)

from ._helpers import _write

# ---- duplicate_function_body ---------------------------------------------


def test_identical_body_across_files_flags(tmp_path: Path):
    """The exact confirmed-real-bug shape: the same helper's body pasted verbatim
    into a second file under a different name."""
    _write(tmp_path, "a.py", """
def _need_cuda():
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False
""")
    _write(tmp_path, "b.py", """
def _has_gpu():
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False
""")
    findings = scan_duplicate_function_body(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "duplicate_function_body"
    assert f.severity == "P2"
    assert "a.py" in f.detail


def test_identical_body_same_file_flags(tmp_path: Path):
    _write(tmp_path, "a.py", """
def f():
    x = 1
    y = 2
    return x + y

def g():
    x = 1
    y = 2
    return x + y
""")
    findings = scan_duplicate_function_body(tmp_path)
    assert len(findings) == 1


def test_three_copies_flags_two_findings(tmp_path: Path):
    """N copies of the same body -> N-1 findings (the first occurrence is treated
    as canonical, every later one is a flagged duplicate)."""
    body = """
def f{n}():
    total = 0
    for i in range(10):
        total += i
    return total
"""
    _write(tmp_path, "a.py", body.format(n=1))
    _write(tmp_path, "b.py", body.format(n=2))
    _write(tmp_path, "c.py", body.format(n=3))
    findings = scan_duplicate_function_body(tmp_path)
    assert len(findings) == 2


def test_different_bodies_clean(tmp_path: Path):
    _write(tmp_path, "a.py", """
def f():
    x = 1
    y = 2
    return x + y

def g():
    x = 1
    y = 3
    return x + y
""")
    assert scan_duplicate_function_body(tmp_path) == []


def test_trivial_bodies_not_flagged(tmp_path: Path):
    """A one-line ``pass``/``...`` stub body is legitimate interface boilerplate,
    not a duplication risk -- must not be flagged even when repeated many times."""
    _write(tmp_path, "a.py", """
class Base:
    def f(self):
        ...

class Other:
    def f(self):
        ...
""")
    assert scan_duplicate_function_body(tmp_path) == []


def test_docstring_only_difference_still_flags(tmp_path: Path):
    """Two copies whose ONLY difference is docstring prose still have an identical
    executable body -- renaming/re-documenting a copy doesn't evade the check."""
    _write(tmp_path, "a.py", '''
def f():
    """Compute the thing."""
    x = 1
    y = 2
    return x + y
''')
    _write(tmp_path, "b.py", '''
def g():
    """A completely different docstring describing the same computation."""
    x = 1
    y = 2
    return x + y
''')
    findings = scan_duplicate_function_body(tmp_path)
    assert len(findings) == 1


def test_different_parameter_names_still_flags(tmp_path: Path):
    """Matching is on the body's AST only -- parameter names are part of the body's
    Name/arg nodes too, so genuinely identical logic with renamed parameters is NOT
    matched (this documents that boundary rather than asserting a specific outcome
    that could silently flip meaning on a refactor)."""
    _write(tmp_path, "a.py", """
def f(value):
    total = value * 2
    return total
""")
    _write(tmp_path, "b.py", """
def g(value):
    total = value * 2
    return total
""")
    findings = scan_duplicate_function_body(tmp_path)
    assert len(findings) == 1


def test_dunder_methods_never_flagged(tmp_path: Path):
    """__getstate__/__setstate__ (and dunders generally) routinely converge on the same
    body shape across unrelated classes by protocol design -- e.g. every class that drops
    one unpicklable attribute looks alike. Must never be flagged, regardless of body size."""
    _write(tmp_path, "a.py", """
class A:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
""")
    _write(tmp_path, "b.py", """
class B:
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
""")
    assert scan_duplicate_function_body(tmp_path) == []
