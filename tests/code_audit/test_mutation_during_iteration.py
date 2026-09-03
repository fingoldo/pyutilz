"""Scanner tests for mutation_during_iteration, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_broad_except_swallows,
    scan_mutation_during_iteration,
)

from ._helpers import _write

# ---- mutation_during_iteration ------------------------------------------


def test_mut_iter_del_dict_during_iter_flags_p0(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(d):
    for k in d:
        if k.startswith("_"):
            del d[k]
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"
    assert findings[0].check == "mutation_during_iteration"


def test_mut_iter_list_append_during_iter_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(items):
    for x in items:
        if cond(x):
            items.append(transform(x))
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"


def test_mut_iter_dict_pop_during_iter_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(d):
    for k, v in d.items():
        if v < 0:
            d.pop(k)
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"


def test_mut_iter_list_copy_pattern_safe(tmp_path: Path):
    """Defensive copy via list(d) is correctly NOT flagged."""
    _write(tmp_path, "ok.py", """
def f(d):
    for k in list(d):
        if k.startswith("_"):
            del d[k]
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert findings == []


def test_mut_iter_copy_method_pattern_safe(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(d):
    for k, v in d.copy().items():
        if cond(v):
            del d[k]
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert findings == []


def test_mut_iter_mutation_on_different_collection_safe(tmp_path: Path):
    """Iterating one collection + mutating a different one is the
    typical correct case."""
    _write(tmp_path, "ok.py", """
def f(src, dst):
    for k in src:
        dst[k] = compute(k)
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert findings == []


def test_mut_iter_assign_existing_key_flags_p1(tmp_path: Path):
    """Reassigning an EXISTING key is size-preserving and safe (CPython),
    but we can't statically tell new vs existing. Flag P1 (lower than
    del/pop) so reviewers can verify."""
    _write(tmp_path, "warn.py", """
def f(d):
    for k in d:
        d[k] = transform(d[k])
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_mut_iter_set_add_during_iter_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(s):
    for x in s:
        if cond(x):
            s.add(transform(x))
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1


def test_broad_except_import_plus_setup_flagged(tmp_path: Path):
    """Import-guard suppression should NOT fire when the try body mixes
    imports with side-effecting setup (the swallow then hides real setup
    failures, not just missing-dep failures)."""
    _write(tmp_path, "bad.py", """
try:
    import torch
    torch.cuda.set_device(0)
except Exception:
    pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "import + side-effect must NOT be allowlisted as pure import guard"
