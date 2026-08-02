"""Tests for pyutilz.dev.code_audit.near_duplicate_function_body.scan_near_duplicate_function_body.

Kept in its own file, separate from the main tests/test_code_audit.py, so it doesn't
collide with unrelated in-flight work in that file.
"""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_near_duplicate_function_body

_NEAR_DUP_PAIR = '''
def process_records(records, threshold):
    results = []
    for r in records:
        if r.get("score", 0) > threshold:
            value = r["score"] * 2
            if value > 100:
                value = 100
            results.append({"id": r["id"], "value": value, "flag": True})
        else:
            results.append({"id": r["id"], "value": 0, "flag": False})
    return results
'''

_NEAR_DUP_DRIFTED = '''
def process_entries(entries, cutoff):
    results = []
    for r in entries:
        if r.get("score", 0) > cutoff:
            value = r["score"] * 2
            if value > 100:
                value = 100
            results.append({"id": r["id"], "value": value, "flag": True})
        else:
            results.append({"id": r["id"], "value": 0, "flag": False})
    return results
'''

_SUBSET_SUPERSET = '''
def process_entries(entries, cutoff):
    results = []
    for r in entries:
        if r.get("score", 0) > cutoff:
            value = r["score"] * 2
            if value > 100:
                value = 100
            results.append({"id": r["id"], "value": value, "flag": True})
        else:
            results.append({"id": r["id"], "value": 0, "flag": False})

    extra_total = 0
    for k in range(50):
        extra_total += k * k
        if extra_total > 10000:
            extra_total -= 5000
        for m in range(10):
            extra_total += m
            if extra_total % 7 == 0:
                extra_total *= 2
    print("done", extra_total, len(results))
    return results
'''

_UNRELATED = '''
def unrelated_function_totally_different(x, y, z):
    total = 0
    for i in range(x):
        for j in range(y):
            total += i * j + z
    return total
'''


def _write(root: Path, name: str, content: str) -> None:
    (root / name).write_text(content, encoding="utf-8")


def test_flags_near_duplicate_pair(tmp_path):
    _write(tmp_path, "a.py", _NEAR_DUP_PAIR)
    _write(tmp_path, "b.py", _NEAR_DUP_DRIFTED)

    findings = scan_near_duplicate_function_body(tmp_path, min_nodes=5)

    near_dup = [f for f in findings if f.check == "near_duplicate_function_body"]
    assert len(near_dup) == 1
    assert near_dup[0].file == "b.py"


def test_flags_subset_containment(tmp_path):
    _write(tmp_path, "a.py", _NEAR_DUP_PAIR)
    _write(tmp_path, "c.py", _SUBSET_SUPERSET)

    findings = scan_near_duplicate_function_body(tmp_path, min_nodes=5)

    subset = [f for f in findings if f.check == "duplicate_function_body_subset"]
    assert len(subset) == 1
    assert subset[0].file == "c.py"
    assert "100%" in subset[0].detail


def test_ignores_unrelated_functions(tmp_path):
    _write(tmp_path, "a.py", _NEAR_DUP_PAIR)
    _write(tmp_path, "d.py", _UNRELATED)

    findings = scan_near_duplicate_function_body(tmp_path, min_nodes=5)

    assert findings == []


def test_ignores_exact_duplicates(tmp_path):
    # An EXACT duplicate is scan_duplicate_function_body's job, not this scanner's --
    # a byte-identical body must not also be reported here.
    _write(tmp_path, "a.py", _NEAR_DUP_PAIR)
    _write(tmp_path, "a_copy.py", _NEAR_DUP_PAIR)

    findings = scan_near_duplicate_function_body(tmp_path, min_nodes=5)

    assert findings == []


def test_respects_min_nodes_floor(tmp_path):
    _write(tmp_path, "a.py", "def f(x):\n    return x + 1\n")
    _write(tmp_path, "b.py", "def g(y):\n    return y + 1\n")

    findings = scan_near_duplicate_function_body(tmp_path, min_nodes=20)

    assert findings == []
