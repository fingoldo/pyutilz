"""Scanner tests for duplicate_conditions, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_duplicate_conditions,
)

from ._helpers import _write

# ---- duplicate_condition -------------------------------------------------


def test_duplicate_or_operand_flags(tmp_path: Path):
    """The exact confirmed-real-bug shape: same endswith suffix twice, the
    intended second suffix silently never checked."""
    _write(tmp_path, "bad.py", """
def f(form):
    if form.endswith('ssions') or form.endswith('ssions'):
        return True
    return False
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_duplicate_and_operand_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(a, b):
    return a > 1 and a > 1
""")
    assert len(scan_duplicate_conditions(tmp_path)) == 1


def test_distinct_operands_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(form):
    if form.endswith('ssions') or form.endswith('ssiez'):
        return True
    return form == "a" or form == "b" or form == "c"
""")
    assert scan_duplicate_conditions(tmp_path) == []


def test_duplicate_elif_test_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(x):
    if x == 1:
        return "a"
    elif x == 2:
        return "b"
    elif x == 1:
        return "dead"
    return "c"
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert len(findings) == 1
    assert "unreachable" in findings[0].detail


def test_distinct_elif_chain_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x):
    if x == 1:
        return "a"
    elif x == 2:
        return "b"
    elif x == 3:
        return "c"
    return "d"
""")
    assert scan_duplicate_conditions(tmp_path) == []


def test_duplicate_elif_not_double_counted_mid_chain(tmp_path: Path):
    """ast.walk visits every If including the elif branches themselves;
    a duplicate between branches 2 and 3 must be reported exactly once
    (only the chain HEAD starts a walk)."""
    _write(tmp_path, "bad.py", """
def f(x):
    if x == 1:
        return "a"
    elif x == 2:
        return "b"
    elif x == 2:
        return "dead"
    return "c"
""")
    assert len(scan_duplicate_conditions(tmp_path)) == 1


def test_separate_if_statements_with_same_test_clean(tmp_path: Path):
    """Two INDEPENDENT if statements (not an elif chain) with the same
    test are legitimate -- state may change between them."""
    _write(tmp_path, "ok.py", """
def f(x, items):
    if x == 1:
        items.append(1)
    if x == 1:
        items.append(2)
    return items
""")
    assert scan_duplicate_conditions(tmp_path) == []


def test_duplicate_dict_key_flags(tmp_path: Path):
    """The exact confirmed-real-bug shape: a correction-table dict
    redefines the same key with a different value 82 lines later,
    silently discarding the first entry (Python keeps only the last)."""
    _write(tmp_path, "bad.py", """
FIXES = {
    "испёк": ("печь", "испечь"),
    "other": ("x", "y"),
    "испёк": ("искать", "испечь"),
}
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "duplicate_dict_key"
    assert findings[0].severity == "P2"


def test_duplicate_dict_key_int_and_bool_alias_flags(tmp_path: Path):
    """1 == True and 0 == False in Python, and they hash equal, so a real
    dict literal collides them too -- the scanner must match that."""
    _write(tmp_path, "bad.py", """
d = {1: "a", True: "b"}
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "duplicate_dict_key"


def test_distinct_dict_keys_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
d = {"a": 1, "b": 2, "c": 3}
""")
    assert scan_duplicate_conditions(tmp_path) == []


def test_dict_key_with_spread_not_crashed(tmp_path: Path):
    """``{**other, "a": 1}`` has a key=None entry for the spread -- must
    not crash comparing None."""
    _write(tmp_path, "ok.py", """
def f(other):
    return {**other, "a": 1, "b": 2}
""")
    assert scan_duplicate_conditions(tmp_path) == []
