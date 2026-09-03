"""Scanner tests for default_via_or, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_default_via_or_trap,
)

from ._helpers import _write

# ---- default_via_or ----------------------------------------------------


def test_default_via_or_int_positive_flags_p1(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(n=None):
    n_jobs = n or 4
    return n_jobs
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1, findings
    assert "or 4" in p1[0].detail


def test_default_via_or_zero_rhs_skipped(tmp_path: Path):
    # `or 0` is a no-op for falsy left -> no real trap.
    _write(tmp_path, "ok.py", """
def f(n=None):
    return n or 0
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1 == []


def test_default_via_or_call_rhs_flags_p2(tmp_path: Path):
    _write(tmp_path, "warn.py", """
def f(data=None):
    return data or compute_default()
""")
    findings = scan_default_via_or_trap(tmp_path)
    p2 = [f for f in findings if f.severity == "P2"]
    assert p2, findings


def test_default_via_or_dict_empty_rhs_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(d=None):
    return d or {}
""")
    findings = scan_default_via_or_trap(tmp_path)
    # `or {}` is exactly the null-safety idiom and is NOT a trap.
    assert all(f.severity != "P1" for f in findings)


def test_default_via_or_os_cpu_count_allowlisted(tmp_path: Path):
    """`os.cpu_count() or 1` is documented-safe (cpu_count returns int or
    None; 0 isn't in the return set), so it must NOT be flagged as a trap."""
    _write(tmp_path, "ok.py", """
import os
def f():
    n = os.cpu_count() or 1
    return n
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1 == [], f"`os.cpu_count() or 1` is documented-safe; got: {p1}"


def test_default_via_or_psutil_cpu_count_allowlisted(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import psutil
def f():
    return psutil.cpu_count(logical=True) or 1
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1 == [], f"`psutil.cpu_count(...) or 1` is documented-safe; got: {p1}"


def test_default_via_or_numpy_std_allowlisted(tmp_path: Path):
    """`np.std(arr) or 1.0` is the intentional divide-by-zero guard
    (std returns 0.0 only when all values equal)."""
    _write(tmp_path, "ok.py", """
import numpy as np
def f(arr):
    return float(np.std(arr)) or 1.0
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1 == [], f"`np.std(arr) or 1.0` is documented-safe; got: {p1}"


def test_default_via_or_len_allowlisted(tmp_path: Path):
    """`len(xs) or N` is the common empty-collection fallback idiom."""
    _write(tmp_path, "ok.py", """
def f(xs):
    return len(xs) or 100
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1 == [], f"`len(xs) or N` is empty-fallback idiom; got: {p1}"


def test_default_via_or_user_attr_still_flagged(tmp_path: Path):
    """User-controlled attribute on the LHS is still flagged: the user
    config may legitimately pass 0 as a sentinel."""
    _write(tmp_path, "bad.py", """
def f(cfg):
    return getattr(cfg, "n_jobs", 1) or 4
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1, "user-attribute LHS must still be flagged"


# ---- default_via_or: boolean-context exclusion (2026-07 large-scale FP fix) ----


def test_default_via_or_if_test_skipped(tmp_path: Path):
    """`if not line or line.startswith(...):` is ordinary control flow,
    not a default-value substitution -- this shape was the single largest
    false-positive class found in a downstream large-scale triage."""
    _write(tmp_path, "ok.py", """
def f(line):
    if not line or line.startswith("#"):
        return None
    return 1
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_elif_test_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x):
    if x == 1:
        return "a"
    elif x == 2 or x == 3:
        return "b"
    return "c"
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_while_test_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x, y):
    while x < 10 or y < 10:
        x += 1
        y += 1
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_assert_test_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x):
    assert x is None or isinstance(x, str)
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_ternary_test_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x, y):
    return "yes" if x == 1 or y == 1 else "no"
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_comprehension_filter_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(items):
    return [i for i in items if i.startswith("a") or i.startswith("b")]
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_nested_boolop_in_if_test_skipped(tmp_path: Path):
    """`(a or b) and c` inside an if-test: the inner Or must still resolve
    to the outer If.test boolean context by climbing through the And."""
    _write(tmp_path, "ok.py", """
def f(a, b, c):
    if (a == 1 or b == 1) and c == 1:
        return 1
    return 0
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_not_wrapped_in_if_test_skipped(tmp_path: Path):
    """`not (a or b)` inside an if-test: the Or must resolve through the
    UnaryOp(Not) wrapper to the outer If.test boolean context."""
    _write(tmp_path, "ok.py", """
def f(a, b):
    if not (a == 1 or b == 1):
        return 1
    return 0
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_assignment_still_flagged_despite_boolean_fix(tmp_path: Path):
    """The boolean-context exclusion must not eat genuine assignment-shape
    findings -- only test/filter positions are exempt."""
    _write(tmp_path, "bad.py", """
def f(cfg):
    x = cfg.get("n") or 7
    return x
""")
    findings = scan_default_via_or_trap(tmp_path)
    p1 = [f for f in findings if f.severity == "P1"]
    assert p1, "assignment-shape `or` must still be flagged after the boolean-context fix"


# ---- default_via_or: 2026-07 precision round 2 --------------------------


def test_default_via_or_inert_falsy_constant_skipped(tmp_path: Path):
    """`x or 0` / `x or ""` cannot corrupt anything: substituting the
    type's own falsy value for a falsy input is observably a no-op.
    233 findings of this shape in a downstream triage -- all benign."""
    _write(tmp_path, "ok.py", """
def f(row, s):
    count = row.get("count") or 0
    score = row.get("score") or 0.0
    label = s or ""
    flag = row.get("flag") or False
    return count, score, label, flag
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_alias_key_get_chain_skipped(tmp_path: Path):
    """`d.get("notes") or d.get("note")` -- substring-related keys are a
    schema-drift alias idiom (canonical vs legacy spelling), not a trap."""
    _write(tmp_path, "ok.py", """
def f(d):
    notes = d.get("notes") or d.get("note")
    ptype = d.get("prosody_type") or d.get("type")
    return notes, ptype
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_non_alias_get_chain_still_flagged(tmp_path: Path):
    """Two DIFFERENT fields chained with `or` must stay flagged -- this
    exact shape (effective vs actual cost) was a confirmed real bug: a
    legitimate $0.00 cached cost silently replaced by the uncached one."""
    _write(tmp_path, "bad.py", """
def f(bundle):
    cost = bundle.get("effective_cost_usd") or bundle.get("actual_cost_usd")
    return cost
""")
    assert scan_default_via_or_trap(tmp_path), "non-alias .get() chain must stay flagged"


def test_default_via_or_constructor_rhs_downgraded_to_low(tmp_path: Path):
    """`x or ClassName()` -- LHS is almost always an `X | None` object
    param (instances always truthy), so this is Low, not P2."""
    _write(tmp_path, "ok.py", """
def f(schedule):
    schedule = schedule or HalvingSchedule()
    return schedule
""")
    findings = scan_default_via_or_trap(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"


def test_default_via_or_lowercase_call_rhs_stays_p2(tmp_path: Path):
    """`x or float("inf")` must stay P2 -- lowercase callables CAN return
    falsy values, and this exact shape was a confirmed real bug (a 0ms
    latency ranked as the worst endpoint)."""
    _write(tmp_path, "bad.py", """
def f(ep):
    latency = ep.get("latency_p50_ms") or float("inf")
    return latency
""")
    findings = scan_default_via_or_trap(tmp_path)
    assert findings and findings[0].severity == "P2"


def test_default_via_or_negative_int_rhs_still_flagged(tmp_path: Path):
    """`scalar() or -1` (UnaryOp RHS) must stay flagged -- this exact
    shape clobbered a legitimate MAX(sense_rank)==0 into -1 in a
    confirmed real bug."""
    _write(tmp_path, "bad.py", """
def f(rank_result):
    next_rank = (rank_result.scalar() or -1) + 1
    return next_rank
""")
    assert scan_default_via_or_trap(tmp_path), "`or -1` must stay flagged"


# ---- default_via_or: 2026-09-03 downstream-scan precision round -------------
#
# A scan of two fresh repos measured 10 of 11 findings as false. Each shape below is one of
# those measured false positives, kept as a negative case.


def test_default_via_or_plain_boolean_locals_skipped(tmp_path: Path):
    """`return is_environ_get or is_getenv` -- two bools stored in locals first. The scanner saw
    two bare Name operands with no default value anywhere and reported a substitution trap."""
    _write(tmp_path, "ok.py", """
def f(node):
    is_environ_get = node.attr == "get"
    is_getenv = node.attr == "getenv"
    return is_environ_get or is_getenv
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_path_exists_predicate_skipped(tmp_path: Path):
    """`(root / t).exists() or any(root.glob(t))` is pure boolean logic: `Path.exists` is
    documented `-> bool` and can never produce a falsy-but-meaningful value."""
    _write(tmp_path, "ok.py", """
def f(root, targets):
    return all((root / t).exists() or any(root.glob(t)) for t in targets)
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_regex_group_alternation_skipped(tmp_path: Path):
    """`m.group("bc") or m.group("test")` reads alternative capture groups: a group that did not
    participate is None, so the chain IS the documented way to ask which alternative matched."""
    _write(tmp_path, "ok.py", """
def f(m):
    return m.group("bc") or m.group("test") or m.group("arith")
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_ast_alias_idiom_skipped(tmp_path: Path):
    """`alias.asname or alias.name` -- the universal import-walking idiom, on its fifth review in
    this project's own baseline. The RHS is a read of another field, not a default value."""
    _write(tmp_path, "ok.py", """
import ast

def f(node):
    return [alias.asname or alias.name for alias in node.names]
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_same_key_other_receiver_skipped(tmp_path: Path):
    """`top.get("context_length") or entry.get("context_length")` -- one field looked up in a
    specific record then in the parent record. Both sides are the same field."""
    _write(tmp_path, "ok.py", """
def f(top, entry):
    return top.get("context_length") or entry.get("context_length")
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_all_caps_constant_rhs_still_flagged(tmp_path: Path):
    """The RHS-is-a-read exclusion must NOT eat `timeout or DEFAULT_TIMEOUT`: the constant-naming
    convention is direct evidence the author is supplying a default, which is the trap itself."""
    _write(tmp_path, "bad.py", """
DEFAULT_TIMEOUT = 30

def f(timeout=None):
    return timeout or DEFAULT_TIMEOUT
""")
    assert scan_default_via_or_trap(tmp_path), "`or DEFAULT_TIMEOUT` must stay flagged"


def test_default_via_or_boolean_flag_parameter_skipped(tmp_path: Path):
    """`guarded or _guard_looks_throttled(test)` -- the left operand is a parameter DECLARED
    `bool`, so the whole expression is boolean logic. Found on this project's own scanners."""
    _write(tmp_path, "ok.py", """
def visit(node, guarded: bool = False):
    child_guarded = guarded or _guard_looks_throttled(node.test)
    return child_guarded
""")
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_same_computed_key_on_two_tables_skipped(tmp_path: Path):
    """`_QWERTY_ADJACENT.get(ch) or _YCUKEN_ADJACENT.get(ch)` -- one lookup table then its
    alternate-layout twin, the same key in both. A non-literal key is the same shape as a literal
    one."""
    _write(tmp_path, "ok.py", """
def neighbours(lower, qwerty, ycuken):
    return qwerty.get(lower) or ycuken.get(lower)
""")
    assert scan_default_via_or_trap(tmp_path) == []
