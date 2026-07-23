"""Unit tests for pyutilz.dev.code_audit AST scanners.

Each scanner gets a positive case (constructed snippet that MUST be
flagged) and a negative case (constructed snippet that MUST NOT be
flagged). Tests use tmp_path so the audit runs against a hermetic
directory; no cross-test bleed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pyutilz.dev.code_audit import (
    Finding,
    run_all,
    scan_broad_except_swallows,
    scan_default_via_or_trap,
    scan_late_binding_closures,
    scan_mutable_defaults,
    scan_mutation_during_iteration,
    scan_nan_equality,
    scan_sql_limit_without_order_by,
    scan_sql_offset_pagination,
    scan_dead_cli_flags,
    scan_log_only_except,
    scan_sql_migration_idempotency,
    scan_duplicate_conditions,
    scan_missed_await,
    scan_redundant_test_fit_calls,
    scan_undeclared_imports,
    scan_vacuous_assertions,
    scan_locals_globals_as_output,
    scan_missing_network_timeout,
    scan_parameter_aliasing_mutation,
    scan_sync_blocking_in_async,
    scan_retry_loops,
    scan_duplicate_module_docstring,
    scan_unraised_exceptions,
    scan_credential_shaped_log_args,
    scan_docstring_args_completeness,
    scan_return_annotation_mismatch,
    scan_sql_aggregate_before_cast,
    scan_locals_get_fragile_lookup,
    scan_shielded_resource_release_race,
    scan_duplicate_credential_regex,
    scan_asymmetric_resource_guard,
    scan_stale_test_spy_arity,
    scan_unthrottled_hot_loop_log,
    scan_possibly_dead_import,
)


def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(source.lstrip("\n"), encoding="utf-8")
    return p


# ---- mutable_default ----------------------------------------------------


def test_mutable_default_mutated_list_flags_p0(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def collect(items=[]):
    items.append(1)
    return items
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "mutable_default"
    assert f.severity == "P0"
    assert "items" in f.detail
    assert "MUTATED" in f.detail


def test_mutable_default_mutated_dict_flags_p0(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def collect(cfg={}):
    cfg["k"] = 1
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"


def test_mutable_default_unmutated_list_flags_low(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def length_only(items=[]):
    return len(items)
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"
    assert "never mutated" in findings[0].detail


def test_mutable_default_call_form_list(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def collect(items=list()):
    items.append(1)
""")
    findings = scan_mutable_defaults(tmp_path)
    assert any(f.severity == "P0" for f in findings)


def test_mutable_default_none_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def collect(items=None):
    items = items if items is not None else []
    items.append(1)
""")
    findings = scan_mutable_defaults(tmp_path)
    assert findings == [], findings


def test_mutable_default_set_form(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def collect(seen=set()):
    seen.add(1)
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1 and findings[0].severity == "P0"


# ---- late_binding_closure ----------------------------------------------


def test_late_binding_lambda_in_for_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def build():
    callbacks = []
    for x in range(5):
        callbacks.append(lambda: x * 2)
    return callbacks
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].severity == "P1"
    assert findings[0].check == "late_binding_closure"


def test_late_binding_lambda_with_default_arg_safe(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def build():
    callbacks = []
    for x in range(5):
        callbacks.append(lambda x=x: x * 2)
    return callbacks
""")
    findings = scan_late_binding_closures(tmp_path)
    assert findings == [], findings


def test_sync_lambda_in_sorted_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def order(items):
    for group in items:
        sorted(group, key=lambda k: group[k])
""")
    # The lambda doesn't escape the iteration (sorted is synchronous).
    findings = scan_late_binding_closures(tmp_path)
    assert findings == []


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


def test_dict_key_non_literal_not_flagged(tmp_path: Path):
    """A computed key (``{x: 1, y: 1}`` where x/y are variables) can't be
    reliably compared statically -- must not false-positive."""
    _write(tmp_path, "ok.py", """
def f(x, y):
    return {x: 1, y: 2}
""")
    assert scan_duplicate_conditions(tmp_path) == []


def test_dict_key_separate_dict_literals_not_conflated(tmp_path: Path):
    """Two separate dict literals reusing the same key are unrelated --
    must not be flagged as a collision within one literal."""
    _write(tmp_path, "ok.py", """
d1 = {"a": 1}
d2 = {"a": 2}
""")
    assert scan_duplicate_conditions(tmp_path) == []


# ---- missed_await ----------------------------------------------------------


def test_missed_await_discarded_coroutine_flags(tmp_path: Path):
    """The true-positive shape: a bare-statement call to a same-module
    async def -- the coroutine is created and discarded, the body never
    runs, and the caller carries on as if the save happened."""
    _write(tmp_path, "bad.py", """
async def do_save(item):
    ...

async def process(item):
    do_save(item)
    return True
""")
    findings = scan_missed_await(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert "do_save" in findings[0].detail


def test_missed_await_from_sync_caller_also_flags(tmp_path: Path):
    """Discarding a coroutine from a SYNC function is the same bug."""
    _write(tmp_path, "bad.py", """
async def notify(msg):
    ...

def handler(msg):
    notify(msg)
""")
    assert scan_missed_await(tmp_path), "sync caller discarding a coroutine must be flagged"


def test_missed_await_awaited_call_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
async def do_save(item):
    ...

async def process(item):
    await do_save(item)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_gather_list_pattern_clean(tmp_path: Path):
    """FP shape #1 from corpus validation: coroutines collected into a
    list and gathered later -- assignment-level calls are never flagged."""
    _write(tmp_path, "ok.py", """
import asyncio

async def work(x):
    ...

async def run(xs):
    tasks = [work(x) for x in xs]
    return await asyncio.gather(*tasks)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_local_import_shadow_clean(tmp_path: Path):
    """FP shape #2 from corpus validation: a function-local import of a
    SYNC function that shares its name with a module-level async def."""
    _write(tmp_path, "ok.py", """
async def count_tokens(text):
    ...

def fallback(text):
    from other_module import count_tokens
    count_tokens(text)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_local_assignment_shadow_clean(tmp_path: Path):
    """FP shape #3: the name is locally rebound to something else."""
    _write(tmp_path, "ok.py", """
async def refresh():
    ...

def run(callbacks):
    refresh = callbacks["refresh"]
    refresh()
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_attribute_call_not_flagged(tmp_path: Path):
    """Attribute calls (self.method(), obj.fn()) are out of scope -- no
    reliable static resolution to a same-module async def."""
    _write(tmp_path, "ok.py", """
class Svc:
    async def ping(self):
        ...

    def run(self):
        self.ping()
""")
    assert scan_missed_await(tmp_path) == []


# ---- broad_except_swallow: precision refinements ----------------------


def test_broad_except_import_guard_skipped(tmp_path: Path):
    """Optional-dep import guards are legitimate broad-except patterns;
    the WHOLE POINT of the swallow is to silently degrade when the dep
    is missing. Don't flag these."""
    _write(tmp_path, "ok.py", """
try:
    import torch
    import torch.nn
except Exception:
    pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"import-guard try-block must not be flagged; got {findings}"


def test_broad_except_import_from_guard_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
try:
    from numba import cuda
except Exception:
    pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_broad_except_best_effort_chmod_skipped(tmp_path: Path):
    """Best-effort filesystem ops (chmod / unlink / makedirs) legitimately
    swallow OSError-class failures."""
    _write(tmp_path, "ok.py", """
import os
def cleanup(path):
    try:
        os.unlink(path)
    except Exception:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"best-effort filesystem op must not be flagged; got {findings}"


def test_broad_except_best_effort_method_skipped(tmp_path: Path):
    """``proc.kill()`` / ``file.close()`` swallows are legitimate."""
    _write(tmp_path, "ok.py", """
def teardown(proc):
    try:
        proc.terminate()
    except Exception:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_broad_except_real_data_swallow_still_flagged(tmp_path: Path):
    """Data-path swallow with non-trivial body MUST still be flagged."""
    _write(tmp_path, "bad.py", """
def process(rows):
    out = []
    for r in rows:
        try:
            out.append(transform(r))
        except Exception:
            continue
    return out
""")
    findings = scan_broad_except_swallows(tmp_path)
    # The try body is a single Call, but it's `out.append(...)` which is
    # in our STORING_METHODS set, not in BEST_EFFORT_OPS. Should still flag.
    assert findings, "data-path swallow with non-best-effort body MUST flag"


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


# ---- broad_except_swallow ----------------------------------------------


def test_broad_except_pass_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        do_thing()
    except Exception:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_broad_except_with_logger_warning_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        do_thing()
    except Exception as exc:
        logger.warning("do_thing failed: %s", exc)
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], findings


def test_broad_except_with_reraise_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        do_thing()
    except Exception:
        cleanup()
        raise
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_broad_except_debug_only_log_clean(tmp_path: Path):
    """A best-effort feature probe that logs at debug level is a genuine
    signal (visible the moment someone enables debug logging) -- not
    equivalent to a truly silent ``except: pass``. This shape was the
    single largest source of false positives in a downstream large-scale
    triage (2026-07): 13 handlers that DID log, just at debug level."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def f():
    try:
        install_optional_filter()
    except Exception as exc:
        logger.debug("Could not install optional filter: %s", exc)
""")
    assert scan_broad_except_swallows(tmp_path) == []


def test_broad_except_no_log_at_all_still_flagged(tmp_path: Path):
    """The debug-only exemption must not widen into a blanket exemption --
    a handler with NO log call whatsoever (any level) is still flagged."""
    _write(tmp_path, "bad.py", """
def f():
    try:
        do_thing()
    except Exception:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "truly silent except: pass must still be flagged"


def test_narrow_except_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        do_thing()
    except KeyError:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_bare_except_pass_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        do_thing()
    except:
        pass
""")
    findings = scan_broad_except_swallows(tmp_path)
    assert len(findings) == 1
    assert "bare except" in findings[0].detail


# ---- sql_limit_without_order_by -----------------------------------------


def test_sql_limit_without_order_by_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
from sqlalchemy import text

def f(session):
    return session.execute(text("""
        SELECT id FROM widgets WHERE flag IS NULL LIMIT :n
    """))
''')
    findings = scan_sql_limit_without_order_by(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_limit_without_order_by"
    assert findings[0].severity == "P2"


def test_sql_limit_with_order_by_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL ORDER BY id LIMIT :n
    """)
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_limit_1_exempted(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("SELECT id FROM widgets LIMIT 1")
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_limit_non_sql_string_ignored(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
LIMIT_MSG = "please stay under the LIMIT of 10 items"
''')
    assert scan_sql_limit_without_order_by(tmp_path) == []


# ---- sql_offset_pagination ------------------------------------------------


def test_sql_offset_pagination_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def f(session, offset):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL
        ORDER BY id LIMIT :n OFFSET :offset
    """)
''')
    findings = scan_sql_offset_pagination(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_offset_pagination"
    assert findings[0].severity == "Low"


def test_sql_limit_without_offset_not_flagged_by_offset_scanner(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(session):
    return session.execute("SELECT id FROM widgets ORDER BY id LIMIT :n")
''')
    assert scan_sql_offset_pagination(tmp_path) == []


def test_sql_offset_pagination_ignores_module_docstring_mentioning_sql_keywords(tmp_path: Path):
    _write(tmp_path, "pkg_init.py", '''
"""Documents this package's scanners.

``scan_sql_offset_pagination``: a SQL literal combining ``LIMIT`` and ``OFFSET``. Advisory --
flags the pattern so a reviewer can confirm the query is a SELECT with a stable filtered set.
"""
''')
    assert scan_sql_offset_pagination(tmp_path) == []
    assert scan_sql_limit_without_order_by(tmp_path) == []


def test_sql_offset_pagination_ignores_class_and_function_docstrings(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
class Foo:
    """A SELECT with LIMIT and OFFSET is discussed here, not executed."""

    def bar(self):
        """Same SELECT/LIMIT/OFFSET vocabulary, still just prose."""
        return 1
''')
    assert scan_sql_offset_pagination(tmp_path) == []


def test_sql_offset_pagination_still_flags_real_sql_after_a_docstring(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
"""This module talks about SELECT, LIMIT and OFFSET in prose."""

def f(session, offset):
    return session.execute("""
        SELECT id FROM widgets WHERE flag IS NULL
        ORDER BY id LIMIT :n OFFSET :offset
    """)
''')
    findings = scan_sql_offset_pagination(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "sql_offset_pagination"


# ---- dead_cli_flag ---------------------------------------------------------


def test_dead_cli_flag_never_read_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    print(args.batch_size)
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "dead_cli_flag"
    assert "resume" in findings[0].detail


def test_cli_flag_read_via_args_attr_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    if args.resume:
        print("resuming")
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_cli_flag_explicit_dest_used(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", dest="resume_flag", action="store_true")
    args = parser.parse_args()
    print(args.resume_flag)
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_cli_flag_used_in_other_file_of_tree_clean(tmp_path: Path):
    _write(tmp_path, "cli_def.py", """
import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    return parser
""")
    _write(tmp_path, "consumer.py", """
def run(args):
    if args.resume:
        pass
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_ignores_selenium_options_add_argument(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in web/browser.py):
    Selenium's ChromeOptions/FirefoxOptions expose an UNRELATED add_argument(flag_string)
    method with the identical name -- it appends a raw command-line flag to a list passed to
    the external Chrome/Firefox binary, with no dest=/action=/etc. concept at all, so
    `.no_sandbox` is never expected to appear anywhere in this codebase's own Python source.
    Distinguished from real argparse usage by the absence of ANY keyword argument."""
    _write(tmp_path, "ok.py", """
from selenium.webdriver.chrome.options import Options

def start_selenium():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=0")
    return options
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_still_flags_argparse_flag_with_a_keyword(tmp_path: Path):
    """The zero-keyword-argument exclusion (added to stop flagging Selenium's unrelated
    add_argument) must not blind the scanner to a genuine dead argparse flag that carries at
    least one argparse-specific keyword -- the shape virtually all real argparse declarations
    use in practice (default=/action=/type=/help=/dest=)."""
    _write(tmp_path, "bad.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False)
    args = parser.parse_args()
    print(args.batch_size)
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert len(findings) == 1
    assert "resume" in findings[0].detail


def test_dead_cli_flag_known_limitation_zero_kwarg_argparse_flag_not_flagged(tmp_path: Path):
    """Documents an accepted trade-off: an argparse flag declared with NO keywords at all
    (bare `add_argument("--resume")`, relying entirely on argparse's defaults) is
    syntactically indistinguishable from Selenium's add_argument and is no longer flagged even
    if genuinely dead. Real argparse declarations in this codebase always carry at least one
    keyword (see dev/code_audit/cli.py), so this is a narrow, low-risk gap traded for
    eliminating a confirmed, concrete false-positive class."""
    _write(tmp_path, "bad_but_unflagged.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(args.batch_size)
""")
    assert scan_dead_cli_flags(tmp_path) == []


# ---- log_only_except -------------------------------------------------------


def test_log_only_except_flags_when_convention_used(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    findings = scan_log_only_except(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "log_only_except"
    assert findings[0].severity == "P2"


def test_log_only_except_escalated_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
        result.validation_errors.append(f"write_failed: {e}")
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_reraise_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
        raise
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_convention_in_file_is_clean(tmp_path: Path):
    """The escalation convention (validation_errors / errors / etc) isn't
    used anywhere in the file, so silence here is a design choice, not a
    detected gap."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def save():
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_log_call_not_double_flagged(tmp_path: Path):
    """No log call at all is scan_broad_except_swallows' territory, not this scanner's."""
    _write(tmp_path, "ok.py", """
def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception:
        pass
""")
    assert scan_log_only_except(tmp_path) == []


# ---- log_only_except: alternate escalation conventions (2026-07 FP fix) ----


def test_log_only_except_error_counter_increment_is_clean(tmp_path: Path):
    """``stats["errors"] += 1`` / ``total_errors += len(batch)`` is a
    legitimate escalation convention this scanner didn't originally
    recognise -- the file's OWN naming (``validation_errors`` elsewhere)
    triggers the file-level scope gate, but the actual handler escalates
    via a differently-shaped counter."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def process(items):
    validation_errors = []
    stats = {"errors": 0}
    for item in items:
        try:
            do_thing(item)
        except Exception as e:
            logger.warning("failed: %s", e)
            stats["errors"] += 1
    return stats
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_return_false_sentinel_is_clean(tmp_path: Path):
    """A Phase0-style ``return False`` on failure is a caller-visible
    escalation contract even though nothing gets appended to a list."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run_test(errors):
    try:
        do_check()
        return True
    except Exception as e:
        logger.warning("check failed: %s", e)
        return False
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_return_error_dict_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors):
    try:
        return {"result": do_thing()}
    except Exception as e:
        logger.warning("failed: %s", e)
        return {"error": str(e)}
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_warn_method_call_is_clean(tmp_path: Path):
    """``results.warn(...)`` -- a distinct object-method escalation
    convention -- is recognised regardless of the base object's name."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors, results):
    try:
        do_thing()
    except Exception as e:
        logger.warning("failed: %s", e)
        results.warn(f"skipped: {e}")
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_local_error_var_assignment_is_clean(tmp_path: Path):
    """Stashing the failure into a local ``error_message``-named variable
    (persisted after the loop) is a real escalation path even without an
    immediate append/return."""
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def run(errors):
    error_message = None
    try:
        do_thing()
    except Exception as e:
        logger.warning("failed: %s", e)
        error_message = str(e)
    return error_message
""")
    assert scan_log_only_except(tmp_path) == []


def test_log_only_except_no_escalation_at_all_still_flagged(tmp_path: Path):
    """None of the recognised escalation conventions apply -- must still
    be flagged (the fix must not become a blanket exemption)."""
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def save(result):
    result.validation_errors = []
    try:
        do_write()
    except Exception as e:
        logger.warning("write failed: %s", e)
""")
    findings = scan_log_only_except(tmp_path)
    assert findings, "handler with no escalation path at all must still be flagged"


# ---- sql_migration_not_idempotent ------------------------------------------


def test_migration_drop_constraint_without_if_exists_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets DROP CONSTRAINT widgets_pkey;\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert "DROP CONSTRAINT" in findings[0].detail


def test_migration_drop_constraint_if_exists_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", "ALTER TABLE widgets DROP CONSTRAINT IF EXISTS widgets_pkey;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_without_if_not_exists_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets ADD COLUMN status TEXT;\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "ADD COLUMN" in findings[0].detail


def test_migration_add_column_if_not_exists_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS status TEXT;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_primary_key_without_do_block_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets ADD PRIMARY KEY (id);\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "PRIMARY KEY" in findings[0].detail


def test_migration_add_primary_key_with_do_block_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", """
DO $$
BEGIN
    ALTER TABLE widgets ADD PRIMARY KEY (id);
END $$;
""")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_in_existence_guarded_do_block_clean(tmp_path: Path):
    """ADD COLUMN inside a DO $$ ... END $$ block that itself probes
    information_schema via IF NOT EXISTS is idempotent at the block level,
    even though the ALTER statement's own line has no IF NOT EXISTS
    keyword -- the classic existence-probe idiom."""
    _write(tmp_path, "ok.sql", """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'widgets' AND column_name = 'status'
    ) THEN
        ALTER TABLE widgets ADD COLUMN status TEXT;
    END IF;
END $$;
""")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_in_plain_do_block_without_guard_flags(tmp_path: Path):
    """A DO block with no existence check at all does not make the ALTER
    inside it idempotent -- must still be flagged."""
    _write(tmp_path, "bad.sql", """
DO $$
BEGIN
    ALTER TABLE widgets ADD COLUMN status TEXT;
END $$;
""")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "ADD COLUMN" in findings[0].detail


def test_migration_non_sql_file_ignored(tmp_path: Path):
    _write(tmp_path, "notes.txt", "ALTER TABLE widgets DROP CONSTRAINT widgets_pkey;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


# ---- run_all + ordering -------------------------------------------------


def test_run_all_returns_sorted_by_severity(tmp_path: Path):
    _write(tmp_path, "mixed.py", """
def bad_mutable(items=[]):
    items.append(1)

def bad_or(n=None):
    return n or 4
""")
    findings = run_all(tmp_path)
    # P0 (mutable_default mutated) should come before P1 (default_via_or).
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=lambda s: {"P0": 0, "P1": 1, "P2": 2, "Low": 3}[s])
    assert "P0" in severities
    assert "P1" in severities


def test_run_all_empty_on_clean_tree(tmp_path: Path):
    _write(tmp_path, "clean.py", """
def f(x=None):
    if x is None:
        x = []
    return x
""")
    findings = run_all(tmp_path)
    assert findings == []


def test_excluded_dir_ignored(tmp_path: Path):
    bad = tmp_path / "build" / "bad.py"
    bad.parent.mkdir()
    bad.write_text("def f(x=[]): x.append(1)\n", encoding="utf-8")
    findings = run_all(tmp_path)
    assert findings == [], "build/ should be excluded by default"


def test_finding_md_row_format():
    f = Finding(
        check="x", severity="P0", file="src/a.py", line=42,
        snippet="def f(x=[])", detail="bad",
    )
    row = f.as_md_row()
    assert row.startswith("| P0 | x | src/a.py:42 |")
    assert "`def f(x=[])`" in row


# ---- CLI surface --------------------------------------------------------


def test_cli_exits_nonzero_on_p1(tmp_path: Path, capsys):
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    from pyutilz.dev.code_audit import main as cli_main
    rc = cli_main([str(tmp_path), "--format", "markdown"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "mutable_default" in out
    assert "P0" in out


def test_cli_exits_zero_on_clean(tmp_path: Path, capsys):
    _write(tmp_path, "ok.py", "def f(x=None):\n    return x\n")
    from pyutilz.dev.code_audit import main as cli_main
    rc = cli_main([str(tmp_path)])
    assert rc == 0


# ---- subpackage facade sensor ------------------------------------------


# ---- redundant_test_fit_call ---------------------------------------------


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_identical_fit_call_across_two_tests_flags(tmp_path: Path):
    """The exact confirmed-real-bug shape (mlframe MRMR biz_value suite): two sibling
    test functions each independently call the SAME deterministic helper with the SAME
    literal seed to check a different assertion on the identical fit result."""
    _write(tmp_path, "test_bad.py", """
def _build_data(seed):
    return seed

def _fit_model(X, seed):
    return X + seed

def test_a():
    X = _build_data(seed=101)
    sel = _fit_model(X, seed=101)
    assert sel

def test_b():
    X = _build_data(seed=101)
    sel = _fit_model(X, seed=101)
    assert sel
""")
    findings = scan_redundant_test_fit_calls(tmp_path)
    checks = {f.check for f in findings}
    assert "redundant_test_fit_call" in checks
    assert all(f.severity == "Low" for f in findings)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_different_seeds_not_flagged(tmp_path: Path):
    """Different literal args -> genuinely different computations, not a duplicate."""
    _write(tmp_path, "test_ok.py", """
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=202)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_same_test_not_flagged(tmp_path: Path):
    """The SAME call appearing twice within one test function (e.g. a sanity re-check)
    is not a cross-test duplication -- only 2+ DIFFERENT test functions count."""
    _write(tmp_path, "test_ok.py", """
def _build_data(seed):
    return seed

def test_a():
    X1 = _build_data(seed=101)
    X2 = _build_data(seed=101)
    assert X1 == X2
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_redundant_call_already_cached_not_flagged(tmp_path: Path):
    """A helper already decorated with @cache/@lru_cache has already been fixed."""
    _write(tmp_path, "test_ok.py", """
from functools import cache

@cache
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_redundant_call_non_test_file_not_scanned(tmp_path: Path):
    """This scanner only applies to test_*.py / *_test.py files."""
    _write(tmp_path, "helpers.py", """
def _build_data(seed):
    return seed

def test_a():
    X = _build_data(seed=101)
    assert X

def test_b():
    X = _build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_redundant_call_public_function_not_flagged(tmp_path: Path):
    """Only underscore-prefixed local helpers are tracked -- repeated identical calls to a
    public/third-party-style function (no leading underscore) are a normal, cheap pattern."""
    _write(tmp_path, "test_ok.py", """
def build_data(seed):
    return seed

def test_a():
    X = build_data(seed=101)
    assert X

def test_b():
    X = build_data(seed=101)
    assert X
""")
    assert scan_redundant_test_fit_calls(tmp_path) == []


def test_facade_reexports_are_same_objects():
    """After the >1000-LOC split into a subpackage, the ``code_audit``
    facade must re-export every public symbol as the SAME object the
    cohesive submodule defines. Guards against a future submodule shuffle
    silently changing the public import surface."""
    import pyutilz.dev.code_audit as facade
    from pyutilz.dev.code_audit._base import Finding as _Finding
    from pyutilz.dev.code_audit.mutable_defaults import scan_mutable_defaults as _smd
    from pyutilz.dev.code_audit.closures import scan_late_binding_closures as _slbc
    from pyutilz.dev.code_audit.default_via_or import scan_default_via_or_trap as _sdvot
    from pyutilz.dev.code_audit.broad_except import scan_broad_except_swallows as _sbes
    from pyutilz.dev.code_audit.nan_equality import scan_nan_equality as _sne
    from pyutilz.dev.code_audit.mutation_during_iteration import scan_mutation_during_iteration as _smdi
    from pyutilz.dev.code_audit.sql_lint import scan_sql_limit_without_order_by as _sslwob, scan_sql_offset_pagination as _ssop
    from pyutilz.dev.code_audit.dead_cli_flags import scan_dead_cli_flags as _sdcf
    from pyutilz.dev.code_audit.silent_escalation import scan_log_only_except as _sloe, DEFAULT_ESCALATION_ATTRS as _DEA
    from pyutilz.dev.code_audit.sql_migrations import scan_sql_migration_idempotency as _ssmi
    from pyutilz.dev.code_audit.duplicate_conditions import scan_duplicate_conditions as _sdc
    from pyutilz.dev.code_audit.missed_await import scan_missed_await as _sma
    from pyutilz.dev.code_audit.redundant_test_fit import scan_redundant_test_fit_calls as _srtfc
    from pyutilz.dev.code_audit.registry import run_all as _ra, SCANNERS as _SCANNERS_CONST
    from pyutilz.dev.code_audit.cli import main as _main

    assert facade.Finding is _Finding
    assert facade.scan_mutable_defaults is _smd
    assert facade.scan_late_binding_closures is _slbc
    assert facade.scan_default_via_or_trap is _sdvot
    assert facade.scan_broad_except_swallows is _sbes
    assert facade.scan_nan_equality is _sne
    assert facade.scan_mutation_during_iteration is _smdi
    assert facade.scan_sql_limit_without_order_by is _sslwob
    assert facade.scan_sql_offset_pagination is _ssop
    assert facade.scan_dead_cli_flags is _sdcf
    assert facade.scan_log_only_except is _sloe
    assert facade.DEFAULT_ESCALATION_ATTRS is _DEA
    assert facade.scan_sql_migration_idempotency is _ssmi
    assert facade.scan_duplicate_conditions is _sdc
    assert facade.scan_missed_await is _sma
    assert facade.scan_redundant_test_fit_calls is _srtfc
    assert facade.run_all is _ra
    assert facade.SCANNERS is _SCANNERS_CONST
    assert facade.main is _main
    # Every scanner in the registry is the facade-level attribute of the same name.
    for fn in facade.SCANNERS.values():
        assert callable(fn)


def test_cli_json_output(tmp_path: Path, capsys):
    _write(tmp_path, "bad.py", "def f(items=[]):\n    items.append(1)\n")
    from pyutilz.dev.code_audit import main as cli_main
    cli_main([str(tmp_path), "--format", "json"])
    import json as _json
    out = capsys.readouterr().out
    payload = _json.loads(out)
    assert isinstance(payload, list)
    assert payload and payload[0]["check"] == "mutable_default"


# --- 2026-07-21 audit regression tests ------------------------------------


def test_cli_min_severity_does_not_weaken_exit_code(tmp_path: Path):
    """Regression: --min-severity previously filtered `findings` BEFORE the exit-code check,
    so a real P1 finding silently exited 0 once filtered out of the display."""
    from pyutilz.dev.code_audit import main as cli_main

    _write(tmp_path, "bad.py", """
async def process(item):
    await item.save()

def caller(item):
    process(item)
""")
    assert cli_main([str(tmp_path), "--min-severity", "Low"]) == 1
    assert cli_main([str(tmp_path), "--min-severity", "P0"]) == 1


def test_mutable_default_not_flagged_when_only_shadowing_nested_func_mutates(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def outer(x=[]):
    def inner(x):
        x.append(1)
        return x
    return inner([1, 2, 3])
""")
    findings = scan_mutable_defaults(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"  # not P0: outer's own x is never mutated


def test_late_binding_closure_flags_list_comprehension(tmp_path: Path):
    _write(tmp_path, "bad.py", """
handlers = [lambda: x for x in range(3)]
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_late_binding_closure_flags_dict_comprehension(tmp_path: Path):
    _write(tmp_path, "bad.py", """
handlers = {i: (lambda: i) for i in range(3)}
""")
    findings = scan_late_binding_closures(tmp_path)
    assert len(findings) == 1


def test_missed_await_not_flagged_when_shadowed_by_nested_def(tmp_path: Path):
    _write(tmp_path, "ok.py", """
async def process(item):
    await item.save()

def sync_wrapper(item):
    def process(x):
        x.touch()
    process(item)
""")
    findings = scan_missed_await(tmp_path)
    assert findings == []


def test_dead_cli_flag_not_flagged_when_read_via_literal_getattr(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
if getattr(args, "resume"):
    print("resuming")
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert findings == []


def test_sql_migration_recognizes_custom_dollar_quote_tag(tmp_path: Path):
    (tmp_path / "migration.sql").write_text(
        """
DO $body$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'pk_users'
    ) THEN
        ALTER TABLE users ADD PRIMARY KEY (id);
    END IF;
END $body$;
""",
        encoding="utf-8",
    )
    findings = scan_sql_migration_idempotency(tmp_path)
    assert findings == []


def test_finding_as_md_row_escapes_pipe_in_detail():
    f = Finding(check="x", severity="Low", file="a.py", line=1, snippet="s", detail="an `X | None` parameter")
    row = f.as_md_row()
    assert "X \\| None" in row
    # Table structure preserved: exactly 4 unescaped pipes delimit the 5 cells (plus outer edges).
    assert row.count("|") - row.count("\\|") == 6


def test_registry_register_scanner_rejects_collision():
    from pyutilz.dev.code_audit.registry import register_scanner, SCANNERS

    def _dummy(root, exclude_dirs=frozenset()):
        return []

    with pytest.raises(ValueError):
        register_scanner("mutable_default", _dummy)
    assert SCANNERS["mutable_default"] is not _dummy

    register_scanner("mutable_default", _dummy, allow_override=True)
    try:
        assert SCANNERS["mutable_default"] is _dummy
    finally:
        register_scanner("mutable_default", scan_mutable_defaults, allow_override=True)


def test_duplicate_conditions_not_flagged_for_impure_bare_function_retry(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    success = attempt() or attempt()
    return success
""")
    findings = scan_duplicate_conditions(tmp_path)
    assert findings == []


def test_nan_equality_ignores_unrelated_dot_nan_attribute(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(result, expected):
    if result.nan == expected.nan:
        return True
    return False
""")
    findings = scan_nan_equality(tmp_path)
    assert findings == []


def test_nan_equality_still_flags_np_nan(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np
def f(x):
    if x == np.nan:
        return True
    return False
""")
    findings = scan_nan_equality(tmp_path)
    assert len(findings) == 1


def test_mutation_during_iteration_list_message_is_backend_agnostic(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(mylist):
    for x in mylist:
        mylist[0] = x * 2
""")
    findings = scan_mutation_during_iteration(tmp_path)
    assert len(findings) == 1
    assert "RuntimeError on dict/set" not in findings[0].detail


# ---- undeclared_import ----------------------------------------------------


def test_undeclared_import_cross_domain_flags_p1(tmp_path: Path):
    (tmp_path / "web").mkdir()
    _write(tmp_path, "web/bad.py", """
import pandas as pd

def f():
    return pd.DataFrame()
""")
    findings = scan_undeclared_imports(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "undeclared_import"
    assert findings[0].severity == "P1"


def test_undeclared_import_own_domain_is_clean(tmp_path: Path):
    (tmp_path / "web").mkdir()
    _write(tmp_path, "web/ok.py", """
import requests

def f():
    return requests.get("http://x", timeout=5)
""")
    findings = scan_undeclared_imports(tmp_path)
    assert findings == []


# ---- vacuous_assertion ------------------------------------------------


def test_vacuous_assertion_bare_true_flagged(tmp_path: Path):
    _write(tmp_path, "test_bad.py", """
def test_thing():
    result = compute()
    assert True
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "vacuous_assertion"


def test_vacuous_assertion_full_domain_or_flagged(tmp_path: Path):
    _write(tmp_path, "test_bad.py", """
def test_thing(result):
    assert result is None or result == {} or isinstance(result, dict)
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert len(findings) == 1


def test_vacuous_assertion_real_check_is_clean(tmp_path: Path):
    _write(tmp_path, "test_ok.py", """
def test_thing():
    result = compute()
    assert result == 42
""")
    findings = scan_vacuous_assertions(tmp_path)
    assert findings == []


# ---- locals_globals_as_output ------------------------------------------


def test_locals_globals_as_output_kwarg_flagged_p1(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def connect(session):
    session.apply(object=locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_locals_globals_as_output_never_passed_to_a_call_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def debug_dump():
    snapshot = locals()
    return snapshot
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []


def test_locals_globals_as_output_read_only_builtin_consumer_is_clean(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in text/strings/__init__.py's
    __dir__()): passing globals()/locals() to a builtin that only ever READS its argument
    (set/list/dict/sorted/len/etc.) is never the "callee writes into it expecting write-back"
    bug this scanner targets."""
    _write(tmp_path, "ok.py", """
def __dir__():
    return sorted(set(globals()))
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []


def test_locals_globals_as_output_still_flags_positional_to_user_function(tmp_path: Path):
    """The read-only-builtin exclusion must not blind the scanner to the real bug shape:
    locals()/globals() passed positionally to a user-defined (non-builtin) function."""
    _write(tmp_path, "bad.py", """
def connect():
    read_config_file(path, locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"


def test_locals_globals_as_output_skips_unparseable_file(tmp_path: Path):
    """A file with a syntax error must be skipped (via _safe_parse returning None), not raise."""
    _write(tmp_path, "broken.py", """
def connect(:
    session.apply(object=locals())
""")
    findings = scan_locals_globals_as_output(tmp_path)
    assert findings == []


# ---- missing_network_timeout -------------------------------------------


def test_missing_network_timeout_flags_bare_get(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import requests

def f():
    return requests.get("http://example.com")
""")
    findings = scan_missing_network_timeout(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "missing_network_timeout"


def test_missing_network_timeout_with_timeout_kwarg_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import requests

def f():
    return requests.get("http://example.com", timeout=5)
""")
    findings = scan_missing_network_timeout(tmp_path)
    assert findings == []


# ---- parameter_aliasing_mutation ---------------------------------------


def test_parameter_aliasing_mutation_flags_p0(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def upsert(history_fields, hash_field):
    returning_fields = history_fields
    returning_fields += [hash_field]
    return returning_fields
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "parameter_aliasing_mutation"
    assert findings[0].severity == "P0"


def test_parameter_aliasing_mutation_copy_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def upsert(history_fields, hash_field):
    returning_fields = history_fields.copy()
    returning_fields += [hash_field]
    return returning_fields
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert findings == []


def test_parameter_aliasing_mutation_immutable_scalar_union_syntax_is_clean(tmp_path: Path):
    """``X | None``-annotated params: += always rebinds (never in-place mutates), so aliasing
    one is not the leak shape this scanner targets."""
    _write(tmp_path, "ok.py", """
def f(total: float | None = None):
    remaining = total
    remaining -= 1.0
    return remaining
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert findings == []


def test_parameter_aliasing_mutation_immutable_scalar_optional_syntax_is_clean(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in
    data/pandaslib/io_ops.py::merge_pickles): ``typing.Optional[X]`` is a Subscript node, not
    the ``X | None`` BinOp shape -- the SAME immutable-scalar guarantee applies to either
    spelling, so both must be recognized for this exemption to actually cover
    typing.Optional-style code (needed for Python < 3.10 compatibility, where ``X | None``
    isn't valid at runtime without ``from __future__ import annotations``)."""
    _write(tmp_path, "ok.py", """
from typing import Optional

def f(sentinel_field: Optional[str] = None):
    current = sentinel_field
    current += "1"
    return current
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert findings == []


def test_parameter_aliasing_mutation_bare_immutable_scalar_annotation_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(total: float):
    remaining = total
    remaining -= 1.0
    return remaining
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert findings == []


def test_parameter_aliasing_mutation_still_flags_mutable_container_despite_annotation(tmp_path: Path):
    """A container-typed (list) parameter must still be flagged -- the immutable-scalar
    exemption must not over-fire onto genuinely mutable types."""
    _write(tmp_path, "bad.py", """
from typing import Optional

def f(items: Optional[list] = None):
    local = items
    local += [1]
    return local
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P0"


def test_parameter_aliasing_mutation_unannotated_param_stays_conservative(tmp_path: Path):
    """No annotation at all -- the type is unknown, so the scanner's conservative default
    (flag the AugAssign) must stay in effect rather than silently assuming immutability."""
    _write(tmp_path, "bad.py", """
def f(x):
    local = x
    local += 1
    return local
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert len(findings) == 1


# ---- sync_blocking_in_async --------------------------------------------


def test_sync_blocking_in_async_flags_bare_requests(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import requests

async def generate():
    return requests.get("http://example.com")
""")
    findings = scan_sync_blocking_in_async(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_sync_blocking_in_async_awaited_httpx_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import httpx

async def generate():
    async with httpx.AsyncClient() as client:
        return await client.get("http://example.com")
""")
    findings = scan_sync_blocking_in_async(tmp_path)
    assert findings == []


# ---- retry_loop ----------------------------------------------------------


def test_retry_loop_busy_loop_flagged_p1(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def connect():
    while True:
        try:
            return do_connect()
        except ConnectionError:
            continue
""")
    findings = scan_retry_loops(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "busy_retry_loop"
    assert findings[0].severity == "P1"


def test_retry_loop_with_sleep_and_break_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import time

def connect():
    while True:
        try:
            result = do_connect()
            break
        except ConnectionError:
            time.sleep(1)
    return result
""")
    findings = scan_retry_loops(tmp_path)
    assert findings == []


def test_retry_loop_sleep_backed_no_break_flagged_low(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import time

def connect():
    while True:
        try:
            return do_connect()
        except ConnectionError:
            time.sleep(1)
""")
    findings = scan_retry_loops(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "unbounded_retry_loop"
    assert findings[0].severity == "Low"


def test_retry_loop_bounded_via_raise_not_break_is_clean(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in
    llm/claude_code_provider.py): a `while True:` retry loop that bounds itself by raising
    once an attempt counter is exceeded (checked BEFORE the loop's own try/except, so nothing
    inside the SAME loop catches it) is just as bounded as one using `break` -- the scanner
    used to only recognize `break`, flagging every raise-bounded retry loop as unbounded."""
    _write(tmp_path, "ok.py", """
import time

def connect(max_attempts=5):
    attempt = 0
    while True:
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError("exceeded max attempts")
        try:
            return do_connect()
        except ConnectionError:
            time.sleep(1)
""")
    findings = scan_retry_loops(tmp_path)
    assert findings == []


# ---- duplicate_module_docstring ------------------------------------------


def test_duplicate_module_docstring_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
"""First docstring."""
"""Second docstring, silently discarded."""

def f():
    pass
''')
    findings = scan_duplicate_module_docstring(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Low"


def test_duplicate_module_docstring_single_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
"""Only docstring."""

def f():
    pass
''')
    findings = scan_duplicate_module_docstring(tmp_path)
    assert findings == []


# ---- unraised_exception_class ---------------------------------------------


def test_unraised_exception_class_never_raised_flagged(tmp_path: Path):
    _write(tmp_path, "exc.py", """
class LLMTruncationError(Exception):
    pass
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "Medium"


def test_unraised_exception_class_raised_in_different_file_is_clean(tmp_path: Path):
    _write(tmp_path, "exc.py", """
class RetryableError(Exception):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import RetryableError

def f():
    raise RetryableError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert findings == []


# ---- credential_shaped_log_arg ---------------------------------------------


def test_credential_shaped_log_arg_unredacted_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def f(proxy):
    logger.info(proxy)
""")
    findings = scan_credential_shaped_log_args(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_credential_shaped_log_arg_redacted_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def f(proxy):
    redacted = proxy.split("@")[1]
    logger.info(redacted)
""")
    findings = scan_credential_shaped_log_args(tmp_path)
    assert findings == []


# ---- docstring_args_incomplete ---------------------------------------------


def test_docstring_args_incomplete_missing_param_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def f(a, b):
    """Do a thing.

    Args:
        a: the first thing.
    """
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "docstring_args_incomplete"
    assert "b" in findings[0].detail


def test_docstring_args_incomplete_all_documented_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(a, b):
    """Do a thing.

    Args:
        a: the first thing.
        b: the second thing.
    """
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert findings == []


def test_docstring_args_incomplete_no_args_section_is_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def f(a, b):
    """Do a thing."""
    return a + b
''')
    findings = scan_docstring_args_completeness(tmp_path)
    assert findings == []


# ---- return_annotation_mismatch --------------------------------------------


def test_return_annotation_mismatch_tuple_literal_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(x) -> float:
    if x < 0:
        return (0.0, 1.0)
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_return_annotation_mismatch_bare_return_none_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(x) -> str:
    if not x:
        return
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert len(findings) == 1


def test_return_annotation_mismatch_consistent_scalar_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x) -> float:
    if x < 0:
        return 0.0
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []


def test_return_annotation_mismatch_optional_annotation_is_skipped(tmp_path: Path):
    _write(tmp_path, "ok.py", """
from typing import Optional

def f(x) -> Optional[float]:
    if x < 0:
        return None
    return x
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []


def test_return_annotation_mismatch_nested_function_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(x) -> float:
    def helper():
        return [1, 2, 3]
    return x + len(helper())
""")
    findings = scan_return_annotation_mismatch(tmp_path)
    assert findings == []


# ---- sql_aggregate_before_cast --------------------------------------------


def test_sql_aggregate_before_cast_json_extract_no_cast_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", '''
def latest_count(cur):
    cur.execute("SELECT MAX(data->>'count') FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "sql_aggregate_before_cast"
    assert findings[0].severity == "P2"


def test_sql_aggregate_before_cast_with_cast_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def latest_count(cur):
    cur.execute("SELECT MAX((data->>'count')::int) FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert findings == []


def test_sql_aggregate_before_cast_no_json_extract_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", '''
def latest(cur):
    cur.execute("SELECT MAX(created_at) FROM events")
''')
    findings = scan_sql_aggregate_before_cast(tmp_path)
    assert findings == []


# ---- locals_get_fragile_lookup --------------------------------------------


def test_locals_get_fragile_lookup_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(flag):
    if flag:
        cached_result = compute()
    return locals().get("cached_result", None)
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "locals_get_fragile_lookup"
    assert findings[0].severity == "P1"


def test_globals_get_fragile_lookup_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    return globals().get("some_name", None)
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "locals_get_fragile_lookup"


def test_locals_get_normal_variable_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(flag):
    cached_result = None
    if flag:
        cached_result = compute()
    return cached_result
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert findings == []


def test_locals_dict_other_method_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    return list(locals().keys())
""")
    findings = scan_locals_get_fragile_lookup(tmp_path)
    assert findings == []


# ---- shielded_resource_release_race ---------------------------------------


def test_shielded_resource_release_race_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await asyncio.shield(_do_work())
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "shielded_resource_release_race"
    assert findings[0].severity == "P0"


def test_shielded_resource_release_race_own_resource_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            own_conn = pool.acquire()
            try:
                await own_conn.execute("insert ...")
            finally:
                release_conn(own_conn)
        await asyncio.shield(_do_work())
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert findings == []


def test_shielded_resource_release_race_no_shield_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await _do_work()
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert findings == []


def test_shielded_resource_release_race_custom_release_names(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await asyncio.shield(_do_work())
    finally:
        my_custom_release(conn)
""")
    assert scan_shielded_resource_release_race(tmp_path) == []
    findings = scan_shielded_resource_release_race(tmp_path, release_call_names=frozenset({"my_custom_release"}))
    assert len(findings) == 1


# ---- duplicate_credential_regex -------------------------------------------


def test_duplicate_credential_regex_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import re

_TOKEN_RE = re.compile(r"token=\\\\w+")
""")
    findings = scan_duplicate_credential_regex(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "duplicate_credential_regex"
    assert findings[0].severity == "P2"


def test_duplicate_credential_regex_inside_canonical_module_is_clean(tmp_path: Path):
    _write(tmp_path, "secrets_scrub.py", """
import re

_TOKEN_RE = re.compile(r"token=\\\\w+")
""")
    findings = scan_duplicate_credential_regex(tmp_path, canonical_module_rel_paths=frozenset({"secrets_scrub.py"}))
    assert findings == []


def test_duplicate_credential_regex_non_credential_pattern_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import re

_DATE_RE = re.compile(r"\\\\d{4}-\\\\d{2}-\\\\d{2}")
""")
    findings = scan_duplicate_credential_regex(tmp_path)
    assert findings == []


# ---- asymmetric_resource_guard --------------------------------------------


def test_asymmetric_resource_guard_transaction_flagged(tmp_path: Path):
    """The motivating shape: query_rows() correctly wraps conn.cursor() in a
    transaction; prefetch_resume_cache(), a sibling method of the SAME class,
    performs the identical conn.cursor() call unwrapped."""
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def query_rows(self, conn, sql, params):
        async with conn.transaction():
            cur = conn.cursor(sql, *params)
            return [row async for row in cur]

    async def prefetch_resume_cache(self, conn, sql, params):
        cur = conn.cursor(sql, *params)
        return [row async for row in cur]
""",
    )
    findings = scan_asymmetric_resource_guard(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "asymmetric_resource_guard"
    assert findings[0].severity == "P0"
    assert "conn.cursor" in findings[0].detail
    assert "prefetch_resume_cache" in findings[0].detail
    assert "query_rows" in findings[0].detail


def test_asymmetric_resource_guard_lock_shape_flagged(tmp_path: Path):
    """Bare self._lock context-manager guard shape (not a .transaction() call)."""
    _write(
        tmp_path,
        "storage.py",
        """
class FileStorage:
    async def close(self):
        self._db.execute("PRAGMA optimize")

    async def write(self, row):
        async with self._lock:
            self._db.execute("insert ...")
""",
    )
    findings = scan_asymmetric_resource_guard(tmp_path)
    assert len(findings) == 1, findings
    assert "self._db.execute" in findings[0].detail
    assert "close" in findings[0].detail
    assert "write" in findings[0].detail


def test_asymmetric_resource_guard_consistently_guarded_is_clean(tmp_path: Path):
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def query_rows(self, conn, sql, params):
        async with conn.transaction():
            return conn.cursor(sql, *params)

    async def prefetch_resume_cache(self, conn, sql, params):
        async with conn.transaction():
            return conn.cursor(sql, *params)
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_consistently_unguarded_is_clean(tmp_path: Path):
    """Both methods agree on NOT guarding -- no asymmetry, nothing to flag
    (this scanner only fires when one method demonstrates the correct
    pattern and a sibling doesn't; it never invents a rule from nothing)."""
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def a(self, conn):
        return conn.execute("select 1")

    async def b(self, conn):
        return conn.execute("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_different_classes_not_compared(tmp_path: Path):
    """The same operation-shape guarded in one class and unguarded in an
    UNRELATED class is not a finding -- the whole point is that ONE class's
    own code already demonstrates its own correct pattern."""
    _write(
        tmp_path,
        "storage.py",
        """
class A:
    async def guarded(self, conn):
        async with conn.transaction():
            return conn.cursor("select 1")

class B:
    async def unguarded(self, conn):
        return conn.cursor("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_single_method_never_flagged(tmp_path: Path):
    """A class with only one method touching a given operation-shape has no
    sibling to compare against -- can't be asymmetric by definition."""
    _write(
        tmp_path,
        "storage.py",
        """
class Solo:
    async def only(self, conn):
        return conn.cursor("select 1")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_custom_guard_names(tmp_path: Path):
    _write(
        tmp_path,
        "storage.py",
        """
class Store:
    async def a(self, conn):
        async with conn.my_custom_guard():
            return conn.execute("select 1")

    async def b(self, conn):
        return conn.execute("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []
    findings = scan_asymmetric_resource_guard(tmp_path, guard_call_names=frozenset({"my_custom_guard"}))
    assert len(findings) == 1


# ---- stale_test_spy_arity ------------------------------------------------


def test_stale_test_spy_arity_flagged(tmp_path: Path):
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "stale_test_spy_arity"
    assert findings[0].severity == "P1"


def test_stale_test_spy_arity_matching_arity_is_clean(tmp_path: Path):
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node, memo=None):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_varargs_spy_is_clean(tmp_path: Path):
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(*args):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_unrelated_patch_target_not_matched(tmp_path: Path):
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(a):
    pass

def test_foo():
    with patch("prod_module.other_function", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_attribute_call_form_matched(tmp_path: Path):
    """A production call site using attribute form (obj.build_rows(...)) must be matched by
    short name the same as a bare-Name call site."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

class Caller:
    def run(self):
        self.build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert len(findings) == 1


def test_stale_test_spy_arity_starred_call_arg_skipped_not_counted(tmp_path: Path):
    """A production call site using `*args` unpacking has an unknowable static arg count --
    must be skipped (not crash, not spuriously counted as 0)."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller(args_tuple):
    build_rows(*args_tuple)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []  # no resolvable real call site -> nothing to compare against


def test_stale_test_spy_arity_call_with_unmatchable_func_expr_skipped(tmp_path: Path):
    """A call whose func expression is neither a bare Name nor an Attribute (e.g. the result of
    a subscript or another call) can't be short-name-matched -- must not crash."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller(dispatch_table):
    dispatch_table["build_rows"](1, 2, 3)
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_skips_production_file_with_syntax_error(tmp_path: Path):
    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node):
    pass

def test_foo():
    with patch("prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert len(findings) == 1


def test_stale_test_spy_arity_skips_test_file_with_syntax_error(tmp_path: Path):
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_broken.py", "def f(:\n    pass\n")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []  # broken test file skipped, no crash


def test_stale_test_spy_arity_patch_call_with_no_positional_args_skipped(tmp_path: Path):
    """A patch(...) call with no positional args at all (e.g. patch(target=..., side_effect=...))
    has no target string to resolve -- must be skipped, not crash on an index error."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def spy(tables, cid, node):
    pass

def test_foo():
    with patch(target="prod_module.build_rows", side_effect=spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_side_effect_not_a_bare_name_skipped(tmp_path: Path):
    """side_effect=<a lambda / call expression>, not a bare Name referencing a local def --
    can't resolve to a spy function's own arity, must be skipped."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch

def test_foo():
    with patch("prod_module.build_rows", side_effect=lambda *a: None):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


def test_stale_test_spy_arity_side_effect_name_not_a_local_def_skipped(tmp_path: Path):
    """side_effect references a Name that isn't a local function def in this test file (e.g.
    imported from elsewhere) -- can't inspect its arity, must be skipped, not crash."""
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass

def caller():
    build_rows(1, 2, 3, 4)
""")
    _write(tmp_path, "test_prod_module.py", """
from unittest.mock import patch
from some_helpers import imported_spy

def test_foo():
    with patch("prod_module.build_rows", side_effect=imported_spy):
        pass
""")
    findings = scan_stale_test_spy_arity(tmp_path)
    assert findings == []


# ---- unthrottled_hot_loop_log ---------------------------------------------


def test_unthrottled_hot_loop_log_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def scan(items, log):
    for item in items:
        if item.bad:
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "unthrottled_hot_loop_log"
    assert findings[0].severity == "P2"


def test_unthrottled_hot_loop_log_throttled_guard_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def scan(items, log):
    for item in items:
        if item.bad:
            if _log_throttle("key"):
                log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_modulo_guard_is_clean(tmp_path: Path):
    _write(tmp_path, "ok2.py", """
def scan(items, log):
    for i, item in enumerate(items):
        if i % 100 == 0:
            log.warning("progress %s", i)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_outside_loop_is_clean(tmp_path: Path):
    _write(tmp_path, "ok3.py", """
def scan(item, log):
    if item.bad:
        log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_debug_call_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok4.py", """
def scan(items, log):
    for item in items:
        log.debug("processing %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_while_loop_flagged(tmp_path: Path):
    _write(tmp_path, "bad2.py", """
def scan(get_next, log):
    while True:
        item = get_next()
        if item.bad:
            log.error("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1


def test_unthrottled_hot_loop_log_else_branch_flagged(tmp_path: Path):
    """An unguarded log call in the `else` of an if/else, inside a loop, must still be flagged --
    only the `if`'s own throttle-guarded body is exempt, not its sibling `else`."""
    _write(tmp_path, "bad3.py", """
def scan(items, log):
    for item in items:
        if item.ok:
            pass
        else:
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1


def test_unthrottled_hot_loop_log_attribute_receiver_and_throttle_call_covered(tmp_path: Path):
    """Both the log receiver AND the throttle-check call are attribute access
    (self.log.warning(...), self.limiter.should_throttle(...)) -- exercises the Attribute
    branches of _call_name/_is_log_call, not just the bare-Name ones."""
    _write(tmp_path, "ok5.py", """
class Scanner:
    def scan(self, items):
        for item in items:
            if self.limiter.should_throttle(item):
                self.log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_non_log_named_receiver_not_flagged(tmp_path: Path):
    """A `.warning(...)` call on a receiver whose name doesn't end in log/logger (e.g. a
    warnings-module-shaped object) is out of scope for this scanner -- not every `.warning(...)`
    call is a logger call."""
    _write(tmp_path, "ok6.py", """
def scan(items, notifier):
    for item in items:
        notifier.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_receiver_from_a_call_not_flagged(tmp_path: Path):
    """A `.warning(...)` call whose receiver is itself a Call (e.g. `get_logger().warning(...)`)
    can't be name-matched by this scanner's simple Name/Attribute receiver check -- exercises the
    receiver_name-stays-None fallthrough."""
    _write(tmp_path, "ok7.py", """
def scan(items):
    for item in items:
        get_logger().warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert findings == []


def test_unthrottled_hot_loop_log_skips_file_with_syntax_error(tmp_path: Path):
    """A file with a syntax error must be skipped, not crash the whole scan -- and a sibling
    valid file in the same directory must still be scanned normally."""
    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _write(tmp_path, "bad4.py", """
def scan(items, log):
    for item in items:
        log.error("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == "bad4.py"


def test_unthrottled_hot_loop_log_guard_call_via_subscript_not_a_throttle_hint(tmp_path: Path):
    """The guard's Call func is neither a bare Name nor an Attribute (e.g. a subscripted
    dispatch-table lookup) -- can't name-match it as a throttle hint, so the log call inside
    stays flagged (exercises _call_name's final None fallthrough)."""
    _write(tmp_path, "bad5.py", """
def scan(items, log, checks):
    for item in items:
        if checks["ok"](item):
            log.warning("bad item %s", item)
""")
    findings = scan_unthrottled_hot_loop_log(tmp_path)
    assert len(findings) == 1


# ---- possibly_dead_import --------------------------------------------------


def test_possibly_dead_import_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import os
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "possibly_dead_import"
    assert findings[0].severity == "Low"


def test_possibly_dead_import_bare_name_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import os

def f():
    return os.getcwd()
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_from_import_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
from pathlib import Path

def f():
    return Path(".")
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_aliased_usage_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
import numpy as np

def f():
    return np.array([1, 2, 3])
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_facade_reexport_suppressed_by_corpus_attribute_access(tmp_path: Path):
    """The exact confirmed-real bug class this scanner exists for: `helper` is imported into
    `facade.py` purely to be re-exported, unused within facade.py itself, but consumed elsewhere
    as `facade.helper` -- must NOT be flagged."""
    _write(tmp_path, "facade.py", """
from _impl import helper
""")
    _write(tmp_path, "test_facade.py", """
import facade

def test_it():
    facade.helper()
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_dunder_all_reexport_is_clean(tmp_path: Path):
    _write(tmp_path, "facade.py", """
from _impl import helper

__all__ = ["helper"]
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_underscore_alias_skipped(tmp_path: Path):
    """`import x as _` is a conventional "explicitly discard" marker, not a name meant to be
    referenced -- must not be flagged as a dead import."""
    _write(tmp_path, "mod.py", """
import os as _
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_from_import_underscore_alias_skipped(tmp_path: Path):
    _write(tmp_path, "mod.py", """
from os import path as _
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_star_import_skipped(tmp_path: Path):
    """A star import can't be usage-checked by name -- must not crash or be flagged."""
    _write(tmp_path, "mod.py", """
from os import *
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_skips_file_with_syntax_error(tmp_path: Path):
    _write(tmp_path, "broken.py", "def f(:\n    pass\n")
    _write(tmp_path, "mod.py", """
import os
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert len(findings) == 1
    assert findings[0].file == "mod.py"


def test_possibly_dead_import_no_imports_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", """
def f():
    return 1
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_relative_import_with_no_module_skipped(tmp_path: Path):
    """`from . import x` (ImportFrom with module=None) is a relative package import -- skipped
    rather than crashing on the None module attribute."""
    _write(tmp_path, "mod.py", """
from . import helper
""")
    findings = scan_possibly_dead_import(tmp_path)
    assert findings == []
