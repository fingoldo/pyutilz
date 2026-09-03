"""Scanner tests for nan_equality, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_nan_equality,
)

from ._helpers import _write

# ---- nan_equality ------------------------------------------------------


def test_nan_equality_float_nan_call_flagged(tmp_path: Path):
    """``x == float("nan")`` is always False; must be flagged P0."""
    _write(tmp_path, "bad.py", """
def f(x):
    if x == float("nan"):
        return "missing"
    return "ok"
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"
    assert findings[0].check == "nan_equality"


def test_nan_equality_np_nan_attr_flagged(tmp_path: Path):
    """``x == np.nan`` (attribute form) must be flagged."""
    _write(tmp_path, "bad.py", """
import numpy as np
def f(x):
    return x == np.nan
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1


def test_nan_equality_neq_form_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np
def f(x):
    if x != np.nan:
        return "valid"
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1
    assert "NotEq" in findings[0].detail


def test_nan_equality_nan_on_left_flagged(tmp_path: Path):
    """Reversed form ``np.nan == x`` must also be caught."""
    _write(tmp_path, "bad.py", """
import numpy as np
def f(x):
    return np.nan == x
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1
    assert "left" in findings[0].detail


def test_nan_equality_correct_isnan_clean(tmp_path: Path):
    """``np.isnan(x)`` is the correct idiom; must NOT be flagged."""
    _write(tmp_path, "ok.py", """
import numpy as np
def f(x):
    return np.isnan(x)
""")
    findings = scan_nan_equality(tmp_path)
    assert findings == []


def test_nan_equality_inf_not_flagged(tmp_path: Path):
    """``x == float("inf")`` is well-defined (inf == inf is True), not a bug."""
    _write(tmp_path, "ok.py", """
def f(x):
    return x == float("inf")
""")
    findings = scan_nan_equality(tmp_path)
    assert findings == []


def test_nan_equality_math_nan_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import math
def f(x):
    return x == math.nan
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1
