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
    scan_duplicate_function_body,
    scan_near_duplicate_function_body,
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
    scan_unpicklable_resource_state,
    scan_readonly_to_numpy_mutation,
    scan_bare_except,
    scan_console_unicode,
    scan_mojibake,
    scan_resource_handle_safety,
    scan_todo_hygiene,
    scan_import_cycles,
    scan_tautological_is_not_none_only_tests,
    scan_except_skip_masks_call_under_test,
    scan_uncurated_star_exports,
    scan_dead_public_callables,
    scan_vacuous_empty_pattern_match,
    scan_tautological_guards,
    scan_table_header_row_drift,
    scan_record_field_flow,
    scan_unenforced_docstring_invariants,
    scan_partial_guard_across_siblings,
    scan_inconsistent_filter,
    scan_regex_integer_parse,
    scan_thresholds_below_documented_result,
    scan_hardcoded_absolute_path_in_test,
    scan_async_primitive_reinit_per_call,
    scan_llm_call_missing_max_tokens_cap,
)


from pyutilz.dev.code_audit.source_text_assertions import scan_source_text_assertions
from pyutilz.dev.code_audit.raising_stub_swallowed import scan_raising_stub_swallowed
from pyutilz.dev.code_audit.lazy_log_assertion import scan_lazy_log_assertion
from pyutilz.dev.code_audit.constructor_param_overwritten import scan_constructor_param_overwritten
from pyutilz.dev.code_audit.stats_key_coverage import scan_stats_key_coverage
from pyutilz.dev.code_audit.sentinel_guard_mismatch import scan_sentinel_guard_mismatch
from pyutilz.dev.code_audit.unit_suffix_mismatch import scan_unit_suffix_mismatch
from pyutilz.dev.code_audit.unreachable_import_fallback import scan_unreachable_import_fallback
from pyutilz.dev.code_audit.asymmetric_except_siblings import scan_asymmetric_except_siblings
from pyutilz.dev.code_audit.effect_flag_outside_its_effect import scan_effect_flag_outside_its_effect
from pyutilz.dev.code_audit.guard_decidable_from_constants import scan_guard_decidable_from_constants
from pyutilz.dev.code_audit.count_then_fetch_same_table import scan_count_then_fetch_same_table
from pyutilz.dev.code_audit.accumulator_helper_bypassed import scan_accumulator_helper_bypassed
from pyutilz.dev.code_audit.test_asserts_against_production_constant import scan_test_asserts_against_production_constant
from pyutilz.dev.code_audit.sentinel_cached_as_answer import scan_sentinel_cached_as_answer
from pyutilz.dev.code_audit.sql_selects_unread_column import scan_sql_selects_unread_column
from pyutilz.dev.code_audit.comment_names_missing_symbol import (
    scan_comment_cites_absolute_line,
    scan_comment_names_missing_symbol,
)
from pyutilz.dev.code_audit.docstring_numbers_moved_to_config import scan_docstring_numbers_moved_to_config


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


# ---- near_duplicate_function_body / duplicate_function_body_subset -------


def test_subset_flags_helper_logic_inlined_instead_of_called(tmp_path: Path):
    """A genuine subset hit: the SAME multi-branch logic is copy-pasted into a bigger
    function instead of calling the already-existing helper that has it."""
    _write(tmp_path, "a.py", """
def helper(x, y, z):
    total = 0
    for i in range(x):
        if i % 2 == 0:
            total += i * y
        else:
            total -= i * z
        if total > 1000:
            total -= 500
    return total


def caller(x, y, z, extra):
    total = 0
    for i in range(x):
        if i % 2 == 0:
            total += i * y
        else:
            total -= i * z
        if total > 1000:
            total -= 500
    return total + extra
""")
    findings = scan_near_duplicate_function_body(tmp_path)
    subset = [f for f in findings if f.check == "duplicate_function_body_subset"]
    assert len(subset) == 1, findings


def test_subset_not_flagged_when_both_delegate_to_shared_helper(tmp_path: Path):
    """False-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, e.g. ``safe_execute``/``safe_execute_values`` both calling
    ``basic_db_execute``, ``tune_spec``/``retune_all`` both calling ``_run_spec_tuning``):
    two thin wrappers that both call the SAME already-shared helper necessarily look
    near-identical -- that's the intended DRY shape, not inlined duplicate logic."""
    _write(tmp_path, "a.py", """
def shared_helper(op, statement, data=None, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize=None):
    pass


def do_one(statement, data=None, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize=None):
    return shared_helper("one", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize)


def do_many(statement, data, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize=None, page_size=100):
    return shared_helper("many", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize, page_size=page_size)
""")
    findings = scan_near_duplicate_function_body(tmp_path)
    assert [f for f in findings if f.check == "duplicate_function_body_subset"] == [], findings


def test_subset_not_flagged_for_independent_deprecated_alias_shims(tmp_path: Path):
    """False-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``EnsurePgTableExists``/``ReadTableIntoDic``/``ReadTableIntoDicReversed``):
    independent deprecated-alias shims for DIFFERENT modern functions still look alike
    because they all follow the same documented ``warnings.warn(...); return modern(...)``
    boilerplate -- not one alias's logic copy-pasted into another."""
    _write(tmp_path, "a.py", """
import warnings

def ensure_pg_table_exists(table, key_field_name="name", id_field_name="id", autocreate_id_type_name=None):
    pass

def read_table_into_dict(dict_enums, table, key_field_name="name", condition="", id_field_name="id", autocreate_id_type_name=None):
    pass

def EnsurePgTableExists(sTable, sKeyFieldName="name", sIdFieldName="id", sAutocreateIdTypeName=None):
    warnings.warn("EnsurePgTableExists is deprecated; use ensure_pg_table_exists instead.", DeprecationWarning, stacklevel=2)
    return ensure_pg_table_exists(table=sTable, key_field_name=sKeyFieldName, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName)

def ReadTableIntoDic(dicEnums, sTable, sKeyFieldName="name", sCondition="", sIdFieldName="id", sAutocreateIdTypeName=None):
    warnings.warn("ReadTableIntoDic is deprecated; use read_table_into_dict instead.", DeprecationWarning, stacklevel=2)
    return read_table_into_dict(dict_enums=dicEnums, table=sTable, key_field_name=sKeyFieldName, condition=sCondition, id_field_name=sIdFieldName, autocreate_id_type_name=sAutocreateIdTypeName)
""")
    findings = scan_near_duplicate_function_body(tmp_path)
    assert [f for f in findings if f.check == "duplicate_function_body_subset"] == [], findings


def test_near_duplicate_not_flagged_for_independent_deprecated_alias_shims(tmp_path: Path):
    """The ``near_duplicate_function_body`` shape (ratio-based, comparable-length bodies) has
    the exact same false-positive class as the ``duplicate_function_body_subset`` shape above --
    two independent deprecated-alias shims for DIFFERENT modern functions, near-identical in
    BOTH length and content because they follow the same boilerplate. Confirmed in the wild
    (2026-08-04): pyutilz's own ``ReadTableIntoDic``/``ReadTableIntoDicReversed`` shims cleared
    the ratio threshold (not just containment) on some Python versions -- the exemption was
    applied only in the containment branch, not here."""
    _write(tmp_path, "a.py", """
import warnings

def modern_a(w, x, y, z, q, r):
    pass

def modern_b(w, x, y, z, q, r):
    pass

def LegacyA(w, x, y, z, q, r):
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    return modern_a(w=w, x=x, y=y, z=z, q=q, r=r)

def LegacyB(w, x, y, z, q, r):
    warnings.warn("deprecated", DeprecationWarning, stacklevel=2)
    return modern_b(w=w, x=x, y=y, z=z, q=q, r=r)
""")
    findings = scan_near_duplicate_function_body(tmp_path)
    assert [f for f in findings if f.check == "near_duplicate_function_body"] == [], findings


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


def test_dead_cli_flag_version_action_not_flagged(tmp_path: Path):
    """--version, action="version" is a universal argparse idiom: the built-in action
    prints and exits internally, application code never reads args.version -- must
    never be flagged regardless of how common this exact shape is."""
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    parser.parse_args()
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_help_action_not_flagged(tmp_path: Path):
    """A manual add_argument("--info", action="help") is the same self-handling
    shape as the built-in -h/--help, just under a different flag name."""
    _write(tmp_path, "ok2.py", """
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--info", action="help")
    parser.parse_args()
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


def test_run_all_parallel_matches_sequential(tmp_path: Path):
    """run_all(parallel=True) (the default) must return the EXACT same findings as
    parallel=False -- scanners are independent (each only appends to its own list), so
    distributing them across a ProcessPoolExecutor must be a pure wall-clock optimization,
    never a behavior change."""
    _write(tmp_path, "mixed.py", """
def bad_mutable(items=[]):
    items.append(1)

def bad_or(n=None):
    return n or 4

try:
    risky()
except Exception:
    pass
""")
    parallel = run_all(tmp_path, parallel=True)
    sequential = run_all(tmp_path, parallel=False)
    assert parallel == sequential
    assert len(parallel) > 0


class TestWorkerPoolSizing:
    """Each worker pays a fixed spawn + import + full-corpus-re-parse cost that does NOT
    shrink as workers are added, so the pool must be sized off PHYSICAL cores and a minimum
    batch of scanners per worker -- not off ``os.cpu_count()`` (logical) or the raw scanner
    count. Sizing it off the logical count measurably made the scan SLOWER (see
    ``_MIN_SCANNERS_PER_WORKER``'s sweep); these pin the fix so it cannot silently regress."""

    def _workers_for(self, n_scanners: int) -> int:
        from pyutilz.dev.code_audit.registry import _MIN_SCANNERS_PER_WORKER, _physical_cpu_count

        return max(2, min(_physical_cpu_count(), n_scanners // _MIN_SCANNERS_PER_WORKER))

    def test_physical_count_does_not_exceed_logical(self):
        import os

        from pyutilz.dev.code_audit.registry import _physical_cpu_count

        physical = _physical_cpu_count()
        assert physical >= 1
        assert physical <= (os.cpu_count() or 1), "physical cores cannot exceed logical CPUs"

    def test_worker_count_never_reaches_one_per_scanner(self):
        """The pre-fix formula was min(len(selected), os.cpu_count()), which on a big machine
        spawned an interpreter per scanner. Every worker beyond the batch threshold adds a
        whole corpus re-parse for a shrinking slice of scan work."""
        for n in (20, 49, 200):
            assert self._workers_for(n) <= n // 5, f"{n} scanners spawned too many workers"

    def test_worker_count_is_capped_by_physical_cores(self):
        from pyutilz.dev.code_audit.registry import _physical_cpu_count

        assert self._workers_for(10_000) == _physical_cpu_count()

    def test_small_scanner_sets_still_get_at_least_two_workers(self):
        """The floor keeps a small-but-parallel run (>= _MIN_SCANNERS_FOR_PARALLEL scanners)
        from degenerating into a single-worker pool, which would be strictly worse than the
        sequential path it already opted out of."""
        assert self._workers_for(4) == 2


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
def test_a_loop_variable_argument_is_not_a_repeated_call(tmp_path: Path):
    """Two functions each looping over their own data unparse to the same signature - `_resolves(x)` -
    while sharing no call at all. The check is for a deterministic repeat of ONE call, and a loop variable
    takes a different value every iteration."""
    _write(tmp_path, "test_loop.py", """
def _resolves(x):
    return bool(x)

def test_a():
    assert all(_resolves(x) for x in [1, 2, 3])

def test_b():
    bad = [x for x in [4, 5] if not _resolves(x)]
    assert not bad
""")
    assert [f for f in scan_redundant_test_fit_calls(tmp_path) if f.check == "redundant_test_fit_call"] == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_module_level_constant_argument_is_still_a_repeated_call(tmp_path: Path):
    """The exemption must key on ITERATION binding, not on "the argument is a name": a constant passed by
    name really is the same value in both callers, which is the genuine duplicate this check exists for."""
    _write(tmp_path, "test_const.py", """
SEED = 101

def _fit(seed):
    return seed

def test_a():
    assert _fit(SEED)

def test_b():
    assert _fit(SEED) == 101
""")
    assert "redundant_test_fit_call" in {f.check for f in scan_redundant_test_fit_calls(tmp_path)}


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_literal_data_factory_is_not_flagged(tmp_path: Path):
    """A helper that fills in a dict literal is the opposite of the expensive fit this scanner hunts: it
    costs microseconds, and its result is a FRESH MUTABLE object each caller then edits. Acting on the
    finding here - caching it, or sharing one fixture - would hand every test the same dict and let one
    test's mutation reach another, so a flag on this shape recommends a bug."""
    _write(tmp_path, "test_factory.py", """
def _item(**over):
    base = {"name": "x", "hits": 13, "denominator": 142}
    base.update(over)
    return base

def test_a():
    assert _item()["hits"] == 13

def test_b():
    assert _item()["denominator"] == 142
""")
    assert [f for f in scan_redundant_test_fit_calls(tmp_path) if f.check == "redundant_test_fit_call"] == []


@pytest.mark.skipif(sys.version_info < (3, 9), reason="scan_redundant_test_fit_calls needs ast.unparse (python>=3.9)")
def test_a_helper_with_no_calls_is_still_flagged(tmp_path: Path):
    """The exemption must key on BUILDING a literal, not merely on containing no expensive-looking call:
    `def _build_data(seed): return seed` is what an expensive builder is reduced to in a scanner test."""
    _write(tmp_path, "test_stub.py", """
def _build_data(seed):
    return seed

def test_a():
    assert _build_data(seed=101) == 101

def test_b():
    assert _build_data(seed=101) is not None
""")
    assert "redundant_test_fit_call" in {f.check for f in scan_redundant_test_fit_calls(tmp_path)}


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
    from pyutilz.dev.code_audit.registry import run_all as _ra, get_scanners as _get_scanners
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
    assert facade.get_scanners is _get_scanners
    # The mutable registry itself is NOT part of the facade -- only the read-only accessor is.
    assert not hasattr(facade, "SCANNERS")
    assert facade.main is _main
    # Every scanner in the registry is the facade-level attribute of the same name.
    for fn in facade.get_scanners().values():
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
    from pyutilz.dev.code_audit.registry import register_scanner, get_scanners

    def _dummy(root, exclude_dirs=frozenset()):
        return []

    with pytest.raises(ValueError):
        register_scanner("mutable_default", _dummy)
    assert get_scanners()["mutable_default"] is not _dummy

    register_scanner("mutable_default", _dummy, allow_override=True)
    try:
        assert get_scanners()["mutable_default"] is _dummy
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


def test_unraised_exception_class_base_covered_by_raised_subclass_is_clean(tmp_path: Path):
    """2026-08-03 glossum audit: a base class's error-signaling contract fires
    whenever ANY in-tree subclass is raised -- `except BaseError:` still
    catches a raised `SpecificError(BaseError)`. Flagging the never-directly-
    raised ancestor here was a false positive (`GlossumError`/`ProviderError`
    in glossum's exceptions.py, never raised directly, only via subclasses
    `JobLockConflictError`/`LLMProviderError` respectively)."""
    _write(tmp_path, "exc.py", """
class BaseError(Exception):
    pass

class SpecificError(BaseError):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import SpecificError

def f():
    raise SpecificError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    assert findings == []


def test_unraised_exception_class_base_without_raised_subclass_still_flagged(tmp_path: Path):
    """The base-class exemption above must not blanket-suppress an unrelated
    sibling base whose OWN subclasses are also never raised."""
    _write(tmp_path, "exc.py", """
class BaseError(Exception):
    pass

class SpecificError(BaseError):
    pass

class OtherBaseError(Exception):
    pass

class OtherSpecificError(OtherBaseError):
    pass
""")
    _write(tmp_path, "use.py", """
from exc import SpecificError

def f():
    raise SpecificError("boom")
""")
    findings = scan_unraised_exceptions(tmp_path)
    flagged = {f.snippet for f in findings}
    assert any("OtherBaseError" in s for s in flagged)
    assert any("OtherSpecificError" in s for s in flagged)
    assert not any("BaseError" in s and "OtherBaseError" not in s for s in flagged)
    assert not any("SpecificError" in s and "OtherSpecificError" not in s for s in flagged)


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


def test_unthrottled_hot_loop_log_while_true_bounded_by_internal_retry_break_is_clean(tmp_path: Path):
    """Real false-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``dev/logginglib.py``'s ``debugged()`` decorator): a `while True:` retry loop
    whose only exit is `if not interactive or attempts >= max_retries: raise` inside the except
    block is EXACTLY as bounded as the already-recognized `while attempts < max_retries:` idiom --
    the bound just lives in an internal break/raise condition instead of the loop's own
    (constant, uninformative) test. Must stay clean; the sibling test above
    (`test_unthrottled_hot_loop_log_while_loop_flagged`) confirms a genuinely-unbounded
    `while True:` loop with no such internal bound is still flagged."""
    _write(tmp_path, "retry.py", """
def call_with_retry(func, log, max_retries=3):
    attempts = 0
    while True:
        try:
            return func()
        except Exception as e:
            log.exception(e)
            attempts += 1
            if attempts >= max_retries:
                raise
""")
    assert scan_unthrottled_hot_loop_log(tmp_path) == []


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


def test_possibly_dead_import_future_annotations_skipped(tmp_path: Path):
    """`from __future__ import annotations` is a compiler directive, never referenced as a
    name by design -- must never be flagged."""
    _write(tmp_path, "mod.py", """
from __future__ import annotations
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_noqa_line_skipped(tmp_path: Path):
    """A line already carrying `# noqa` has already been reviewed and explicitly exempted --
    re-flagging it is pure noise."""
    _write(tmp_path, "mod.py", """
import os  # noqa: F401
""")
    assert scan_possibly_dead_import(tmp_path) == []
    _write(tmp_path, "mod2.py", """
from os import path  # noqa: F401
""")
    assert scan_possibly_dead_import(tmp_path) == []


def test_possibly_dead_import_multiline_block_reports_each_dead_name_separately(tmp_path: Path):
    """Real bug found dogfooding this scanner on pyutilz's own source (2026-08-04): a multi-line
    `from x import (a, b, c)` block used to report every dead name at the SAME `node.lineno` (the
    opening line of the statement), and the baseline-diff harness keys findings on exactly
    `(check, file, line)` -- so two independently-dead names in the same block collided onto one
    key and one silently masked the other. Confirmed in the wild: `database/db/__init__.py`'s
    multi-line `sql_helpers` import had THREE independently-unused names (`nu`,
    `MakeSetExcludedClause`, `update_if_now`) but only one finding ever surfaced. Each alias must
    be reported at its OWN line so distinct dead names never collide onto one key."""
    _write(tmp_path, "mod.py", """
from helper_module import (
    used_name,
    dead_one,
    dead_two,
)

def f():
    return used_name()
""")
    findings = scan_possibly_dead_import(tmp_path)
    flagged_names = {f.detail.split("binds '")[1].split("'")[0] for f in findings}
    flagged_lines = {f.line for f in findings}
    assert flagged_names == {"dead_one", "dead_two"}, findings
    assert len(flagged_lines) == 2, findings  # each dead name got its OWN line, not a shared one


def test_alias_own_lineno_fallback_when_ast_alias_lacks_lineno():
    """``ast.alias`` only gained ``lineno``/``col_offset`` in Python 3.10 (bpo-39235) -- on 3.8/3.9
    (this package's own supported floor), the getattr(alias, "lineno", node.lineno) fallback used
    to collapse every alias in a multi-line block back onto node.lineno, silently reproducing the
    exact collision test_possibly_dead_import_multiline_block_reports_each_dead_name_separately
    exists to prevent (confirmed failing in CI on Python 3.8, 2026-08-04). This test exercises the
    fallback path directly (source-text line scan) regardless of which Python actually runs it,
    by stripping ``lineno`` off a real parsed alias before calling the helper -- so the fallback's
    correctness doesn't depend on which interpreter happens to run the test suite."""
    import ast

    from pyutilz.dev.code_audit.dead_import import _alias_own_lineno

    src = "from helper_module import (\n    foo,\n    foo_bar,\n)\n"
    tree = ast.parse(src)
    src_lines = src.splitlines()
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)

    class _Py38Alias:
        """Mimics ast.alias on Python <3.10: no lineno/col_offset attributes at all."""

        def __init__(self, real: ast.alias) -> None:
            self.name = real.name
            self.asname = real.asname

    claimed: set[int] = set()
    linenos = [_alias_own_lineno(_Py38Alias(alias), node, src_lines, claimed) for alias in node.names]  # type: ignore[arg-type]

    # Each alias gets its own distinct line, in source order -- including the substring-collision
    # case (foo vs foo_bar) where a naive `name in line` check would wrongly match "foo"'s line
    # against "foo_bar"'s text too.
    assert linenos == [2, 3]
    assert len(set(linenos)) == 2


def test_alias_own_lineno_fallback_skips_comment_line_repeating_the_name():
    """A same-block why-comment documenting an otherwise-flagged import routinely repeats the
    bound name in prose (this project's own convention: "consumed via `from x import foo`"),
    which would satisfy the fallback's name-pattern match on the COMMENT line, several lines
    before the scan ever reaches the real import line -- misattributing the finding. Confirmed
    live in CI on Python 3.9 (2026-08-27): mlframe's discretization/__init__.py and
    hermite_fe/__init__.py both carry exactly this comment style, and the fallback reported
    findings on the comment lines instead of the actual import lines."""
    import ast

    from pyutilz.dev.code_audit.dead_import import _alias_own_lineno

    src = "from helper_module import (\n    # consumed via `from x import foo` by tests/test_x.py\n    foo,\n)\n"
    tree = ast.parse(src)
    src_lines = src.splitlines()
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)

    class _Py38Alias:
        def __init__(self, real: ast.alias) -> None:
            self.name = real.name
            self.asname = real.asname

    claimed: set[int] = set()
    lineno = _alias_own_lineno(_Py38Alias(node.names[0]), node, src_lines, claimed)  # type: ignore[arg-type]

    assert lineno == 3  # the real `foo,` line, not the line-2 comment mentioning "foo"


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


# ---- unpicklable_resource_state ------------------------------------------


def test_unpicklable_resource_state_lock_without_getstate_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import threading

class Cache:
    def __init__(self):
        self._lock = threading.Lock()
        self._mem = {}
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "unpicklable_resource_state"
    assert f.severity == "P2"
    assert "Cache" in f.detail
    assert "_lock" in f.detail


def test_unpicklable_resource_state_bare_rlock_import_flagged(tmp_path: Path):
    """A directly-imported ``RLock`` (not ``threading.RLock()``) must match on the constructor
    name alone, not require the ``threading.`` prefix."""
    _write(tmp_path, "bad.py", """
from threading import RLock

class Guarded:
    def __init__(self):
        self._lock = RLock()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings


def test_unpicklable_resource_state_open_file_handle_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
class LogWriter:
    def __init__(self, path):
        self._fh = open(path, "w")
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    assert "_fh" in findings[0].detail


def test_unpicklable_resource_state_with_getstate_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import threading

class Cache:
    def __init__(self):
        self._lock = threading.Lock()
        self._mem = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == [], f"class with __getstate__ must not be flagged; got {findings}"


def test_unpicklable_resource_state_plain_attribute_not_flagged(tmp_path: Path):
    """A class whose __init__ only assigns plain data (no lock/thread/file) must not be flagged."""
    _write(tmp_path, "ok.py", """
class Config:
    def __init__(self, name):
        self.name = name
        self.values = {}
        self.count = 0
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == []


def test_unpicklable_resource_state_no_init_not_flagged(tmp_path: Path):
    """A class with no __init__ at all must not crash the scanner or be flagged."""
    _write(tmp_path, "ok.py", """
class Bare:
    x = 1
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert findings == []


def test_unpicklable_resource_state_thread_ctor_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import threading

class Worker:
    def __init__(self):
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        pass
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings
    assert "_thread" in findings[0].detail


def test_unpicklable_resource_state_dotted_cuda_stream_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import torch

class GpuBuffer:
    def __init__(self):
        self._stream = torch.cuda.Stream()
""")
    findings = scan_unpicklable_resource_state(tmp_path)
    assert len(findings) == 1, findings


# ---- readonly_to_numpy_mutation -------------------------------------------


def test_readonly_to_numpy_mutation_fill_diagonal_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np

def dataset_diagnostics(df):
    C = df.corr().to_numpy()
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "readonly_to_numpy_mutation"
    assert f.severity == "P2"
    assert "C" in f.detail


def test_readonly_to_numpy_mutation_copy_true_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import numpy as np

def dataset_diagnostics(df):
    C = df.corr().to_numpy(copy=True)
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


def test_readonly_to_numpy_mutation_copyto_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np

def f(df, other):
    C = df.to_numpy()
    np.copyto(C, other)
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert len(findings) == 1, findings


def test_readonly_to_numpy_mutation_fresh_numpy_array_not_flagged(tmp_path: Path):
    """A ``fill_diagonal`` target that never came from ``.to_numpy()`` (e.g. built via
    ``np.corrcoef``) is out of scope for this scanner."""
    _write(tmp_path, "ok.py", """
import numpy as np

def f(X):
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


def test_readonly_to_numpy_mutation_unrelated_to_numpy_not_flagged(tmp_path: Path):
    """An uncopied ``.to_numpy()`` result that is never passed to an in-place mutator is fine."""
    _write(tmp_path, "ok.py", """
def f(df):
    arr = df.to_numpy()
    return arr.sum()
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


# ---- tautological_is_not_none_only_test ----------------------------------


def test_tautological_is_not_none_only_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_bad.py",
        """
def test_returns_something():
    result = compute()
    assert result is not None
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "tautological_is_not_none_only_test"
    assert "test_returns_something" in findings[0].detail


def test_tautological_is_not_none_with_stronger_assert_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_returns_something():
    result = compute()
    assert result is not None
    assert result.value == 42
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


def test_tautological_is_not_none_nested_in_if_not_flagged(tmp_path: Path):
    """A bare is-not-None inside a conditional branch isn't the function's only unconditional
    check -- scanner is conservative and skips nested asserts entirely."""
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_conditional():
    result = compute()
    if result:
        assert result is not None
    assert result.status == "ok"
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


def test_tautological_is_not_none_non_test_function_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def helper():
    result = compute()
    assert result is not None
""",
    )
    findings = scan_tautological_is_not_none_only_tests(tmp_path)
    assert findings == []


# ---- except_skip_masks_call_under_test -----------------------------------


def test_except_skip_masks_real_call_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_bad.py",
        """
import pytest

def test_something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        pytest.skip("environment issue")
    assert result is not None
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "except_skip_masks_call_under_test"


def test_except_skip_import_guard_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
import pytest

def test_something():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


def test_except_no_skip_call_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "test_ok.py",
        """
def test_something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        raise
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


def test_except_skip_non_test_file_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "helper.py",
        """
import pytest

def something():
    try:
        result = train_model(x=1, y=2)
    except Exception:
        pytest.skip("bad")
""",
    )
    findings = scan_except_skip_masks_call_under_test(tmp_path)
    assert findings == []


# ---- uncurated_star_export -------------------------------------------


def test_uncurated_star_export_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "uncurated_star_export"


def test_uncurated_star_export_with_init_all_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *

__all__ = ["public_helper"]
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_uncurated_star_export_with_submodule_all_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def public_helper():
    return 1

__all__ = ["public_helper"]
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from .sub import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_uncurated_star_export_absolute_import_not_flagged(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "__init__.py",
        """
from numpy import *
""",
    )
    findings = scan_uncurated_star_exports(tmp_path)
    assert findings == []


def test_broad_except_nosec_comment_on_except_line_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def probe():
    try:
        import cupy as cp
        return cp, True
    except Exception:  # nosec B110 - GPU probe is opportunistic, CPU fallback below
        return None, False
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"nosec-documented swallow must not be flagged; got {findings}"


def test_broad_except_opportunistic_keyword_in_handler_body_skipped(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def probe(n_full, n_sub):
    try:
        from ._gpu import fast_path
        return fast_path(n_full, n_sub)
    except Exception:
        # GPU path is opportunistic; any failure falls through to the host path below.
        pass
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == [], f"opportunistic-documented swallow must not be flagged; got {findings}"


def test_broad_except_best_effort_keyword_hyphenated_and_spaced_both_match(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def a():
    try:
        risky()
    except Exception:
        pass  # best-effort cleanup, safe to skip

def b():
    try:
        risky()
    except Exception:
        pass  # best effort cleanup, safe to skip
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings == []


def test_broad_except_unrelated_nosec_elsewhere_in_function_does_not_exempt(tmp_path: Path):
    """The exemption window is the handler's own line + body span -- an unrelated nosec comment on
    a DIFFERENT, unrelated line elsewhere in the same function must not accidentally exempt a real,
    undocumented swallow."""
    _write(
        tmp_path,
        "bad.py",
        """
def f(rows):
    eval(rows)  # nosec B307 - trusted internal input, unrelated to the block below
    out = []
    try:
        out.append(transform(rows))
    except Exception:
        continue
    return out
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "an unrelated nosec comment elsewhere in the function must not suppress a real finding"


def test_broad_except_no_rationale_still_flagged(tmp_path: Path):
    """Sanity: a plain undocumented swallow with none of the rationale markers is still flagged --
    confirms the new exemption isn't accidentally matching everything."""
    _write(
        tmp_path,
        "bad.py",
        """
def process(rows):
    out = []
    for r in rows:
        try:
            out.append(transform(r))
        except Exception:
            continue
    return out
""",
    )
    findings = scan_broad_except_swallows(tmp_path)
    assert findings, "undocumented swallow must still be flagged"


def test_default_via_or_boolean_valued_return_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def overlaps(lo_a, hi_a, lo_b, hi_b):
    return not (hi_a < lo_b or hi_b < lo_a)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == [], f"pure-boolean return must not be flagged; got {findings}"


def test_default_via_or_isinstance_or_isinstance_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def is_not_or_ne(op):
    return isinstance(op, int) or isinstance(op, float)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_startswith_endswith_not_flagged(tmp_path: Path):
    """``str.startswith``/``str.endswith`` always return a real bool, never an arbitrary
    falsy value -- same class of false positive already fixed for ``.all()``/``.any()``."""
    _write(
        tmp_path,
        "ok.py",
        """
def is_diminutive(lemma):
    return lemma.endswith("chen") or lemma.endswith("lein")
""",
    )
    assert scan_default_via_or_trap(tmp_path) == []


def test_default_via_or_looks_and_has_predicate_names_not_flagged(tmp_path: Path):
    """``_foo_looks_bar(...)`` / ``_foo_has_bar(...)`` follow the same predicate-shaped-name
    convention as ``is_*``, just spelled differently -- both sides here are ``-> bool``."""
    _write(
        tmp_path,
        "ok.py",
        """
def _loop_looks_bounded_retry(test):
    return True


def _loop_body_has_meaningful_sleep(stmts):
    return True


def check(test, stmts):
    return _loop_looks_bounded_retry(test) or _loop_body_has_meaningful_sleep(stmts)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_underscore_prefixed_is_predicate_not_flagged(tmp_path: Path):
    """A private helper's leading underscore (``_is_known_immutable_scalar_annotation``) must not
    defeat the ``is_*`` predicate-name recognition -- module-privacy doesn't change the naming
    convention's meaning."""
    _write(
        tmp_path,
        "ok.py",
        """
def _is_known_immutable_scalar_annotation(x):
    return True


def check(a, b):
    return _is_known_immutable_scalar_annotation(a) or _is_known_immutable_scalar_annotation(b)
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_empty_tuple_default_not_flagged(tmp_path: Path):
    """``x or ()`` is the tuple-literal spelling of the same trivial-empty-container idiom already
    covered for ``[]``/``{}``/``set()`` -- empty in, empty out, no distinct value to clobber."""
    _write(
        tmp_path,
        "ok.py",
        """
def normalize(items):
    for g in items or ():
        pass
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_boolean_valued_assignment_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def check(a, b):
    ok = (a > 0) or (b > 0)
    return ok
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings == []


def test_default_via_or_non_boolean_return_still_flagged(tmp_path: Path):
    """Sanity: a genuine default-via-or trap in a return statement is still caught -- the new
    exemption only suppresses PURE-boolean operands, not arbitrary return-position ors."""
    _write(
        tmp_path,
        "bad.py",
        """
def get_count(x):
    return x.count or 5
""",
    )
    findings = scan_default_via_or_trap(tmp_path)
    assert findings, "a non-boolean-valued or-default in a return must still be flagged"


def test_possibly_dead_import_facade_reexport_consumed_via_from_import_elsewhere(tmp_path: Path):
    """A name re-exported by a package __init__.py, consumed elsewhere ONLY via
    `from package import name` (never as `package.name` attribute access), must not be flagged."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def do_thing():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from pkg.sub import do_thing
""",
    )
    consumer_dir = tmp_path / "consumer"
    consumer_dir.mkdir()
    _write(
        consumer_dir,
        "user.py",
        """
from pkg import do_thing

do_thing()
""",
    )
    findings = scan_possibly_dead_import(tmp_path)
    assert findings == [], f"facade re-export consumed via a downstream from-import must not be flagged; got {findings}"


def test_possibly_dead_import_facade_reexport_never_imported_anywhere_still_flagged(tmp_path: Path):
    """Sanity: a name imported into __init__.py but genuinely never consumed anywhere (no bare-name
    use, no attribute access, no downstream from-import) must still be flagged."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _write(
        pkg,
        "sub.py",
        """
def do_thing():
    return 1
""",
    )
    _write(
        pkg,
        "__init__.py",
        """
from pkg.sub import do_thing
""",
    )
    findings = scan_possibly_dead_import(tmp_path)
    assert any(f.file.endswith("__init__.py") for f in findings), "a genuinely unconsumed re-export must still be flagged"


def test_log_only_except_nosec_documented_not_flagged(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
def process(rows):
    validation_errors = []
    try:
        risky()
    except Exception as e:  # nosec B110 - opportunistic path, logging is sufficient here
        logger.warning("failed: %s", e)
""",
    )
    findings = scan_log_only_except(tmp_path)
    assert findings == [], f"nosec-documented log-only except must not be flagged; got {findings}"


# ---- dead_public_callable ------------------------------------------------


def test_dead_public_callable_flags_a_function_only_tests_call(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _write(src, "mod.py", """
def used_by_entry():
    return 1


def measured_but_dead():
    return 2


def main():
    return used_by_entry()
""")
    findings = scan_dead_public_callables(src)
    assert [f.detail.split("'")[1] for f in findings] == ["measured_but_dead"], findings


def test_dead_public_callable_respects_consumer_roots_and_decorators(tmp_path: Path):
    src = tmp_path / "src"
    demo = tmp_path / "demo"
    src.mkdir()
    demo.mkdir()
    _write(src, "mod.py", """
import functools


def called_from_demo():
    return 1


@functools.lru_cache
def framework_invoked():
    return 2
""")
    _write(demo, "run.py", """
from mod import called_from_demo

print(called_from_demo())
""")
    assert scan_dead_public_callables(src, consumer_roots=(demo,)) == []


# ---- vacuous_empty_pattern_match ----------------------------------------


def test_vacuous_empty_pattern_match_flags_unguarded_all(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def covers(text, stems):
    return all(stem in text for stem in stems)
""")
    findings = scan_vacuous_empty_pattern_match(tmp_path)
    assert len(findings) == 1 and findings[0].check == "vacuous_empty_pattern_match", findings


def test_vacuous_empty_pattern_match_accepts_a_guard_and_ignores_any(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def covers(text, stems):
    return bool(stems) and all(stem in text for stem in stems)


def touches(text, stems):
    return any(stem in text for stem in stems)
""")
    assert scan_vacuous_empty_pattern_match(tmp_path) == []


# ---- tautological_guard --------------------------------------------------


def test_tautological_guard_flags_threshold_anded_with_identity_pin(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def label(causes, lead):
    for c in causes:
        if c.mean >= 0.75 * lead and c is causes[0]:
            return "strongly supported"
    return "weak"
""")
    findings = scan_tautological_guards(tmp_path)
    assert len(findings) == 1 and findings[0].check == "tautological_guard", findings


def test_tautological_guard_ignores_none_checks_and_distinct_targets(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def label(a, b, lead):
    if a.mean >= 0.75 * lead and a is not None:
        return 1
    if a.mean >= lead and b is lead:
        return 2
    return 0
""")
    assert scan_tautological_guards(tmp_path) == []


# ---- table_header_row_drift ---------------------------------------------


def test_table_header_row_drift_flags_dictwriter_key_mismatch(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import csv


def dump(fh, rows):
    writer = csv.DictWriter(fh, fieldnames=["a", "b", "c"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"a": row[0], "b": row[1]})
""")
    findings = scan_table_header_row_drift(tmp_path)
    assert any(f.severity == "P1" for f in findings), findings


def test_table_header_row_drift_accepts_matching_keys(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import csv


def dump(fh, rows):
    writer = csv.DictWriter(fh, fieldnames=["a", "b"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"a": row[0], "b": row[1]})
""")
    assert scan_table_header_row_drift(tmp_path) == []


# ---- record_field_flow ---------------------------------------------------


def test_record_field_flow_flags_defaulted_read_of_an_unwritten_key(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def build(hedge):
    return {"mechanism_modality_source": hedge}


def weight(triple):
    return triple.get("mechanism_modality_sources", "unstated")
""")
    findings = scan_record_field_flow(tmp_path)
    assert [f.check for f in findings if f.severity == "P1"] == ["field_read_never_written"], findings


def test_record_field_flow_near_miss_only_ignores_a_foreign_schema_key(tmp_path: Path):
    """A key of somebody else's JSON resembles nothing this tree writes, and must not be reported."""
    _write(tmp_path, "client.py", """
def parse(response):
    return response.get("esearchresult", {})
""")
    assert scan_record_field_flow(tmp_path) == []
    assert scan_record_field_flow(tmp_path, near_miss_only=False), "the exhaustive form must still see it"


def test_record_field_flow_ignores_a_key_with_both_sides(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def build():
    return {"modality_source": "code"}


def read(row):
    return row.get("modality_source", "")
""")
    assert scan_record_field_flow(tmp_path) == []


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


# ---- partial_guard_across_siblings / inconsistent_filter -----------------


def test_partial_guard_across_siblings_flags_the_odd_one_out(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def score_a(rows):
    if not rows:
        return 0.0
    return len(rows)


def score_b(rows):
    if not rows:
        return 0.0
    return len(rows) * 2


def score_c(rows):
    return len(rows) * 3
""")
    findings = scan_partial_guard_across_siblings(tmp_path)
    assert len(findings) == 1 and "score_c" in findings[0].detail, findings


def test_inconsistent_filter_is_silent_until_configured(tmp_path: Path):
    _write(tmp_path, "mod.py", """
def rank(graph):
    return graph.causes()
""")
    assert scan_inconsistent_filter(tmp_path) == []
    findings = scan_inconsistent_filter(tmp_path, filter_pairs=(("causes", "postmortem_events"),))
    assert len(findings) == 1 and "rank" in findings[0].detail, findings


# ---- measurement hygiene -------------------------------------------------


def test_regex_integer_parse_truncation_flags_a_bare_digit_class(tmp_path: Path):
    _write(tmp_path, "bad.py", r'''
import re


def count(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0
''')
    findings = scan_regex_integer_parse(tmp_path)
    assert len(findings) == 1, findings


def test_regex_integer_parse_truncation_accepts_a_real_number_pattern(tmp_path: Path):
    _write(tmp_path, "ok.py", r'''
import re


def count(text):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(m.group().replace(",", "")) if m else 0.0
''')
    assert scan_regex_integer_parse(tmp_path) == []


def test_threshold_below_documented_result_flags_a_weakened_gate(tmp_path: Path):
    _write(tmp_path, "test_cards.py", '''
def test_cards_recover_the_expected_cause():
    """7 of 8 demonstration cards recover the expected cause."""
    decided = 7
    assert decided >= 6
''')
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings


def test_threshold_below_documented_result_accepts_a_matching_gate(tmp_path: Path):
    _write(tmp_path, "test_cards.py", '''
def test_cards_recover_the_expected_cause():
    """7 of 8 demonstration cards recover the expected cause."""
    decided = 7
    assert decided >= 7
''')
    assert scan_thresholds_below_documented_result(tmp_path) == []


# --- field/text agreement: a structured field vs the free text that duplicates it ---------------------


def _temporal_rule():
    """A representative FieldTextRule: the forensic temporal_class pair, cues and anti-cues included."""
    from pyutilz.dev.code_audit import FieldTextRule

    return FieldTextRule(
        name="temporal_class",
        field="temporal_class",
        text_fields=("subject", "object"),
        cues={"antemortem": ("antemortem", "vital", "vitality"), "postmortem": ("postmortem", "putrefaction")},
        anti_cues={"antemortem": ("vital organs",)},
        neutral_values=frozenset({"na", ""}),
        partitions=(frozenset({"antemortem", "perimortem", "agonal"}), frozenset({"postmortem", "artifact"})),
    )


def test_field_text_agreement_flags_a_field_its_own_text_contradicts():
    from pyutilz.dev.code_audit import CONTRADICT, KIND_OPPOSED, check_record

    v = check_record(_temporal_rule(), {"subject": "haemorrhage", "object": "vital hanging", "temporal_class": "postmortem"})
    assert v.outcome == CONTRADICT and v.kind == KIND_OPPOSED and v.supported == "antemortem"


def test_field_text_agreement_reads_a_neutral_field_as_unfilled_not_as_agreement():
    from pyutilz.dev.code_audit import CONTRADICT, KIND_UNFILLED, check_record

    v = check_record(_temporal_rule(), {"subject": "x", "object": "ante-mortem hanging", "temporal_class": "na"})
    assert v.outcome == CONTRADICT and v.kind == KIND_UNFILLED


def test_field_text_agreement_hyphenation_does_not_hide_a_cue():
    from pyutilz.dev.code_audit import cues_in_text

    rule = _temporal_rule()
    assert cues_in_text(rule, "ante-mortem hanging") == cues_in_text(rule, "antemortem hanging")


def test_field_text_agreement_anti_cue_cancels_only_the_homograph():
    from pyutilz.dev.code_audit import AGREE, UNCHECKABLE, check_record

    rule = _temporal_rule()
    assert check_record(rule, {"subject": "injury to vital organs", "object": "y", "temporal_class": "perimortem"}).outcome == UNCHECKABLE
    # ...and the same word still fires where it is a real vitality claim.
    assert check_record(rule, {"subject": "vital reaction", "object": "y", "temporal_class": "antemortem"}).outcome == AGREE


def test_field_text_agreement_compatible_partition_members_agree():
    from pyutilz.dev.code_audit import AGREE, CONTRADICT, check_record

    rule = _temporal_rule()
    assert check_record(rule, {"subject": "vital reaction", "object": "y", "temporal_class": "perimortem"}).outcome == AGREE
    assert check_record(rule, {"subject": "putrefaction", "object": "y", "temporal_class": "artifact"}).outcome == AGREE
    assert check_record(rule, {"subject": "putrefaction", "object": "y", "temporal_class": "antemortem"}).outcome == CONTRADICT


def test_field_text_agreement_publishes_coverage_and_an_empty_vocabulary_is_uncheckable():
    from pyutilz.dev.code_audit import FieldTextRule, check_records

    rows = [{"subject": "a", "object": "b", "temporal_class": "na"}, {"subject": "putrefaction", "object": "b", "temporal_class": "postmortem"}]
    rep = check_records(_temporal_rule(), rows)
    assert (rep.agree, rep.contradict, rep.uncheckable) == (1, 0, 1)
    assert rep.coverage == 0.5 and rep.as_dict()["coverage"] == 0.5
    blank = check_records(FieldTextRule(name="manner", field="manner", text_fields=("object",)), rows)
    assert blank.uncheckable == 2 and blank.agree == 0 and blank.has_vocabulary is False


def test_field_text_agreement_resolver_overrides_the_cue_table():
    from pyutilz.dev.code_audit import CONTRADICT, FieldTextRule, check_record

    rule = FieldTextRule(
        name="modality",
        field="modality",
        text_fields=("quote",),
        neutral_values=frozenset({"unstated", ""}),
        resolver=lambda rec: ("may", "possibly") if "possibly" in str(rec.get("quote", "")) else ("", ""),
    )
    assert check_record(rule, {"quote": "possibly fatal", "modality": "usual"}).outcome == CONTRADICT
    assert check_record(rule, {"quote": "fatal", "modality": "usual"}).outcome == "uncheckable"


def test_field_text_agreement_renders_a_finding():
    from pyutilz.dev.code_audit import Finding, check_record

    v = check_record(_temporal_rule(), {"subject": "x", "object": "vital hanging", "temporal_class": "postmortem"})
    f = v.as_finding(file="bench/gold/PMC13161347.json", line=1)
    assert isinstance(f, Finding) and f.check == "field_text_temporal_class" and "postmortem" in f.detail


# --- domain boundary: domain vocabulary inside code declared domain-neutral --------------------


_BOUNDARY_SOURCE = '''
"""A module holding one neutral concept and one domain concept, not yet split."""


class Envelope:
    """Per-assertion lineage: who observed it, where it is written down."""

    source_id: str = ""
    quote: str = ""


def pool(observations):
    """Pool independent observations of one claim."""
    return sum(observations)


def rank_causes_of_death(rows):
    """Rank the autopsy findings by how well they explain the decedent."""
    return sorted(rows)
'''


def _boundary_tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    _write(tmp_path, "pkg/envelope.py", _BOUNDARY_SOURCE)
    return tmp_path


def test_domain_vocabulary_leak_is_silent_until_configured(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    assert scan_domain_vocabulary_leak(root) == []
    assert scan_domain_vocabulary_leak(root, boundary=[BoundarySymbol("pkg/envelope.py", "pool")], vocabulary=[]) == []
    assert scan_domain_vocabulary_leak(root, boundary=[], vocabulary=["autopsy"]) == []


def test_domain_vocabulary_leak_passes_a_clean_boundary_symbol(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "Envelope"), BoundarySymbol("pkg/envelope.py", "pool")],
        vocabulary=["autopsy", "decedent", "postmortem"],
    )
    assert findings == []


def test_domain_vocabulary_leak_flags_a_term_in_a_docstring_of_a_boundary_symbol(tmp_path: Path):
    """The leak that matters most is prose: a docstring is where a reader learns what the code is about."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "rank_causes_of_death", note="ranking is neutral")],
        vocabulary=["autopsy", "decedent"],
    )
    assert {f.check for f in findings} == {"domain_vocabulary_leak"}
    assert sorted(f.detail.split("domain term ")[1].split(" ")[0] for f in findings) == ["'autopsy'", "'decedent'"]
    assert "ranking is neutral" in findings[0].detail
    # Two leaks in ONE symbol must stay distinguishable after a ratchet truncates the detail, or the
    # second term would be silently absorbed by the first one's baseline entry.
    assert len({f.detail[:110] for f in findings}) == 2


def test_domain_vocabulary_leak_ignores_the_domain_outside_the_boundary(tmp_path: Path):
    """A term in a sibling symbol is not a leak: the boundary is the claim, not the file."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "pool")],
        vocabulary=["autopsy", "decedent"],
    )
    assert findings == []


def test_domain_vocabulary_leak_honours_an_allowed_term(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    kwargs = dict(boundary=[BoundarySymbol("pkg/envelope.py", "rank_causes_of_death")], vocabulary=["autopsy", "decedent"])
    assert len(scan_domain_vocabulary_leak(root, **kwargs)) == 2
    assert len(scan_domain_vocabulary_leak(root, allowed=["decedent"], **kwargs)) == 1


def test_domain_vocabulary_leak_matches_on_word_boundaries_not_substrings(tmp_path: Path):
    """`death` must not fire on `deathless` -- a substring rule would make the vocabulary unusable."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    _write(tmp_path, "sub.py", '''
def pool(x):
    """A deathless abstraction."""
    return x
''')
    findings = scan_domain_vocabulary_leak(tmp_path, boundary=[BoundarySymbol("sub.py", "pool")], vocabulary=["death"])
    assert findings == []


def test_domain_boundary_reports_a_stale_manifest_rather_than_passing_by_vacuity(tmp_path: Path):
    """A renamed symbol must fail loudly: a boundary that names nothing passes for the wrong reason."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "pool_renamed_away"), BoundarySymbol("pkg/gone.py", "anything")],
        vocabulary=["autopsy"],
    )
    assert [f.check for f in findings] == ["boundary_symbol_missing", "boundary_symbol_missing"]
    assert all(f.severity == "P1" for f in findings)


def test_domain_vocabulary_leak_reaches_a_method_of_a_class(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    _write(tmp_path, "meth.py", '''
class Store:
    def neutral(self, x):
        return x

    def pool(self, rows):
        """Pool over autopsy series."""
        return rows
''')
    boundary = [BoundarySymbol("meth.py", "Store.neutral"), BoundarySymbol("meth.py", "Store.pool")]
    findings = scan_domain_vocabulary_leak(tmp_path, boundary=boundary, vocabulary=["autopsy"])
    assert [(f.check, f.line) for f in findings] == [("domain_vocabulary_leak", 6)]


def test_getattr_unknown_attribute_catches_a_printer_reading_a_field_that_never_existed(tmp_path):
    """The regression this rule was written for, reduced to its shape.

    A demonstration script printed two headline panels as empty because it asked one dataclass for `steps`
    and another for `lines` - neither had ever existed - and `getattr(obj, name, None) or []` swallowed both.
    The work behind the panels was computed and paid for on every run, and strategy was argued from a blank.
    """
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n"
        "\n"
        "@dataclass\n"
        "class Sheet:\n"
        "    ask: list = field(default_factory=list)\n"
        "    notes: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "def show(sheet):\n"
        "    for row in getattr(sheet, 'steps', None) or []:\n"
        "        print(row)\n"
        "    for row in getattr(sheet, 'ask', None) or []:\n"
        "        print(row)\n",
        encoding="utf-8",
    )

    findings = scan_getattr_unknown_attribute(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 2)]
    assert "'steps'" in findings[0].detail
    assert findings[0].severity == "P1"


def test_getattr_unknown_attribute_does_not_fire_on_names_the_tree_uses_as_attributes(tmp_path):
    """A name assigned as an attribute anywhere is evidence it exists - including on objects we do not define.

    `threading.local()` and plain namespace objects gain their attributes by assignment and by nothing else,
    so a rule that only read class bodies would report every such lookup as a miss.
    """
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "state.py").write_text(
        "import threading\n"
        "\n"
        "CACHE_DIR = '/tmp'\n"
        "_local = threading.local()\n"
        "\n"
        "def open_db():\n"
        "    _local.handle = object()\n"
        "\n"
        "def read(module):\n"
        "    a = getattr(_local, 'handle', None)\n"
        "    b = getattr(module, 'CACHE_DIR', None)\n"
        "    return a, b\n",
        encoding="utf-8",
    )
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_does_not_fire_on_module_level_def_or_import_bindings(tmp_path):
    """Real false-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``performance/kernel_tuning/cache/cache_base.py``): a common facade-patchability
    idiom is ``getattr(some_module, "func_name", func_name)`` -- looking a name up on a LIVE
    module object (so a test's ``monkeypatch.setattr(module, "func_name", ...)`` is honored) with
    the in-tree function/import as the fallback. The module-level-bindings widening this scanner
    already documents ("since `getattr(some_module, 'NAME', default)` is a legitimate pattern")
    only walked module-level `Assign`/`AnnAssign` though, missing `def`/`class`/`import` bindings
    entirely -- both a module-level function AND a module-level `from x import y` name must count
    as known."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "facade.py").write_text(
        "import sys\n"
        "from math import sqrt\n"
        "\n"
        "def _probe() -> int:\n"
        "    return 1\n"
        "\n"
        "class Widget:\n"
        "    pass\n"
        "\n"
        "def use_facade():\n"
        "    _facade = sys.modules[__name__]\n"
        "    probe = getattr(_facade, '_probe', _probe)\n"
        "    root = getattr(_facade, 'sqrt', sqrt)\n"
        "    widget_cls = getattr(_facade, 'Widget', Widget)\n"
        "    return probe(), root(4), widget_cls()\n",
        encoding="utf-8",
    )
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_ignores_the_two_argument_form(tmp_path):
    """A two-argument getattr raises on a miss, which is loud. The default is what makes the mistake silent."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "loud.py").write_text("def f(o):\n    return getattr(o, 'nowhere')\n", encoding="utf-8")
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_accepts_out_of_tree_names(tmp_path):
    """`extra_known` is how a project states that an attribute belongs to a class it does not define."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "client.py").write_text("def f(provider):\n    return getattr(provider, 'last_generation_id', '')\n", encoding="utf-8")
    assert len(scan_getattr_unknown_attribute(tmp_path)) == 1
    assert scan_getattr_unknown_attribute(tmp_path, extra_known=frozenset({"last_generation_id"})) == []


def test_getattr_literal_on_known_dataclass_catches_a_field_that_belongs_to_a_different_class(tmp_path):
    """Sharper than scan_getattr_unknown_attribute: `steps` is a real field, just not on `Sheet`.

    The union-based rule would miss this because `steps` IS an attribute of something in the tree
    (`Plan`); only per-function local type-tracking catches that the object actually being read
    from is the wrong class for that name.
    """
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n"
        "\n"
        "@dataclass\n"
        "class Sheet:\n"
        "    ask: list = field(default_factory=list)\n"
        "    notes: list = field(default_factory=list)\n"
        "\n"
        "@dataclass\n"
        "class Plan:\n"
        "    steps: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show():\n" "    sheet = Sheet()\n" "    for row in getattr(sheet, 'steps', None) or []:\n" "        print(row)\n",
        encoding="utf-8",
    )

    findings = scan_getattr_literal_on_known_dataclass(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 5)]
    assert "'steps'" in findings[0].detail and "Sheet" in findings[0].detail


def test_getattr_literal_on_known_dataclass_does_not_fire_on_its_own_real_field(tmp_path):
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show():\n" "    sheet = Sheet()\n" "    return getattr(sheet, 'ask', None)\n",
        encoding="utf-8",
    )
    assert scan_getattr_literal_on_known_dataclass(tmp_path) == []


def test_getattr_literal_on_known_dataclass_does_not_fire_when_the_type_cannot_be_inferred(tmp_path):
    """Duck-typing across an intentional boundary is the escape hatch, not a false positive to fix."""
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "def show(sheet):\n" "    return getattr(sheet, 'steps', None)\n",
        encoding="utf-8",
    )
    assert scan_getattr_literal_on_known_dataclass(tmp_path) == []


def test_getattr_literal_on_known_dataclass_infers_type_from_a_parameter_annotation(tmp_path):
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show(sheet: Sheet):\n" "    return getattr(sheet, 'steps', None)\n",
        encoding="utf-8",
    )
    findings = scan_getattr_literal_on_known_dataclass(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 4)]


# ---- bare_except -----------------------------------------------------------


def test_bare_except_bare_colon_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        risky()
    except:
        pass
""")
    findings = scan_bare_except(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "bare_except"


def test_bare_except_base_exception_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f():
    try:
        risky()
    except BaseException:
        pass
""")
    findings = scan_bare_except(tmp_path)
    assert len(findings) == 1, findings


def test_bare_except_base_exception_reraise_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        risky()
    except BaseException:
        cleanup()
        raise
""")
    assert scan_bare_except(tmp_path) == []


def test_bare_except_narrow_exception_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f():
    try:
        risky()
    except Exception:
        pass
""")
    assert scan_bare_except(tmp_path) == []


# ---- console_unicode --------------------------------------------------------


def test_console_unicode_print_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", 'print("done → next")\n')
    findings = scan_console_unicode(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "console_unicode"


def test_console_unicode_logger_call_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", 'logger.warning("bad ✓ value")\n')
    findings = scan_console_unicode(tmp_path)
    assert len(findings) == 1, findings


def test_console_unicode_ascii_only_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", 'print("done -> next")\n')
    assert scan_console_unicode(tmp_path) == []


def test_console_unicode_non_console_call_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", 'save_to_file("→")\n')
    assert scan_console_unicode(tmp_path) == []


def test_console_unicode_stdout_reconfigure_suppresses_file(tmp_path: Path):
    """A file that already forces UTF-8 stdio at its own entry point can't hit the
    UnicodeEncodeError this scanner exists to catch -- confirmed as this codebase's own
    established fix (dozens of scripts use exactly this idiom)."""
    _write(
        tmp_path,
        "ok.py",
        'import sys\nsys.stdout.reconfigure(encoding="utf-8", errors="replace")\nprint("done → next")\n',
    )
    assert scan_console_unicode(tmp_path) == []


def test_console_unicode_stderr_reconfigure_suppresses_file(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        'import sys\nsys.stderr.reconfigure(encoding="utf-8")\nlogger.warning("bad ✓ value")\n',
    )
    assert scan_console_unicode(tmp_path) == []


def test_console_unicode_reconfigure_without_encoding_kwarg_does_not_suppress(tmp_path: Path):
    """A bare ``sys.stdout.reconfigure()`` (no ``encoding=``, e.g. line-buffering tweaks) doesn't
    change the console encoding -- must not be mistaken for the UTF-8 fix."""
    _write(
        tmp_path,
        "bad.py",
        'import sys\nsys.stdout.reconfigure(line_buffering=True)\nprint("done → next")\n',
    )
    findings = scan_console_unicode(tmp_path)
    assert len(findings) == 1, findings


def test_console_unicode_package_init_reconfigure_suppresses_submodule(tmp_path: Path):
    """A package's own __init__.py reconfiguring stdio protects every module beneath it --
    the guard fires on the package's FIRST import regardless of which submodule/entry point
    actually runs, so a submodule needs no guard of its own."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", 'import sys\nsys.stdout.reconfigure(encoding="utf-8")\n')
    _write(pkg, "sub.py", 'def f():\n    print("done → next")\n')
    assert scan_console_unicode(tmp_path) == []


def test_console_unicode_nested_package_without_init_reconfigure_still_flagged(tmp_path: Path):
    """A sibling package with NO reconfiguring __init__.py anywhere in its own chain must still
    be flagged -- the exemption only follows the actual package containment chain, not the
    whole scanned tree."""
    protected = tmp_path / "protected"
    protected.mkdir()
    _write(protected, "__init__.py", 'import sys\nsys.stdout.reconfigure(encoding="utf-8")\n')
    unprotected = tmp_path / "unprotected"
    unprotected.mkdir()
    _write(unprotected, "__init__.py", '"""No reconfigure here."""\n')
    _write(unprotected, "sub.py", 'def f():\n    print("done → next")\n')
    findings = scan_console_unicode(tmp_path)
    assert len(findings) == 1 and "unprotected/sub.py" in findings[0].file, findings


# ---- mojibake ---------------------------------------------------------------


def test_mojibake_roundtrip_corruption_flagged(tmp_path: Path):
    corrupted = "Русский".encode().decode("cp1251")
    _write(tmp_path, "bad.py", f"# {corrupted}\nx = 1\n")
    findings = scan_mojibake(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "mojibake"


def test_mojibake_genuine_cyrillic_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# Привет мир\nx = 1\n")
    assert scan_mojibake(tmp_path) == []


def test_mojibake_ascii_only_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# a normal comment\nx = 1\n")
    assert scan_mojibake(tmp_path) == []


def test_mojibake_short_cyrillic_regex_range_not_flagged(tmp_path: Path):
    # Real false positive found while dogfooding this scanner on pyutilz itself
    # (src/pyutilz/text/humanizer.py): a regex character class like [A-ZА-ЯЁ] gets split
    # by the ASCII "-" into a short 2-char Cyrillic run ("ЯЁ") that coincidentally
    # round-trips through cp1251-encode -> utf-8-decode into different, legible-looking
    # text -- purely by chance, not because anything is actually corrupted.
    _write(tmp_path, "ok.py", 'hits = [m.start() for m in re.finditer(r"\\. [A-ZА-ЯЁ]", text)]\n')
    assert scan_mojibake(tmp_path) == []


# ---- resource_handle_safety --------------------------------------------------


def test_resource_handle_safety_bare_open_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
def f(path):
    fh = open(path, "w")
    fh.write("x")
""")
    findings = scan_resource_handle_safety(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "resource_handle_safety"


def test_resource_handle_safety_popen_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import subprocess

def f():
    proc = subprocess.Popen(["ls"])
    return proc
""")
    findings = scan_resource_handle_safety(tmp_path)
    assert len(findings) == 1, findings


def test_resource_handle_safety_with_block_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
def f(path):
    with open(path, "w") as fh:
        fh.write("x")
""")
    assert scan_resource_handle_safety(tmp_path) == []


# ---- todo_hygiene -------------------------------------------------------------


def test_todo_hygiene_unattributed_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", "# TODO: handle empty list case\nx = 1\n")
    findings = scan_todo_hygiene(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "todo_hygiene"


def test_todo_hygiene_dated_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# TODO 2026-04-28: handle empty list case\nx = 1\n")
    assert scan_todo_hygiene(tmp_path) == []


def test_todo_hygiene_assignee_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# TODO(alice): handle empty list case\nx = 1\n")
    assert scan_todo_hygiene(tmp_path) == []


# ---- import_cycle -------------------------------------------------------------


def test_import_cycles_two_node_cycle_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "import mypkg.a\n")
    findings = scan_import_cycles(pkg, package_name="mypkg")
    assert len(findings) == 1, findings
    assert findings[0].check == "import_cycle"


def test_import_cycles_acyclic_not_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "x = 1\n")
    assert scan_import_cycles(pkg, package_name="mypkg") == []


def test_import_cycles_deferred_cycle_not_flagged(tmp_path: Path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "import mypkg.a\n")
    findings = scan_import_cycles(pkg, package_name="mypkg", deferred_cycles=frozenset({"mypkg.a -> mypkg.b"}))
    assert findings == []


def test_import_cycles_lazy_function_body_import_not_flagged(tmp_path: Path):
    """A cycle that only closes via a lazy (function-body) import is not a module-load-time cycle."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    _write(pkg, "__init__.py", "")
    _write(pkg, "a.py", "import mypkg.b\n")
    _write(pkg, "b.py", "def f():\n    import mypkg.a\n    return mypkg.a\n")
    assert scan_import_cycles(pkg, package_name="mypkg") == []


def test_import_cycles_scc_representative_deterministic_regardless_of_graph_build_order():
    """``scan_import_cycles`` picks ``comp[0]`` (the first element of the SCC Tarjan's algorithm
    returns) as the representative for BOTH ``Finding.file`` and ``Finding.snippet`` -- so which
    element that is must not depend on the ORDER ``_build_graph`` happened to insert modules/
    edges in (which in production tracks ``Path.rglob()``'s filesystem-enumeration order, not
    guaranteed stable across process runs on every OS/filesystem for a large multi-directory
    tree). Confirmed by direct construction: the SAME 4-node cycle, fed to
    ``_strongly_connected_components`` with its dict/edge-set built in forward vs. reversed
    insertion order, previously returned a DIFFERENT first element (Tarjan's DFS visits `graph[v]`
    -- a plain ``set`` -- in insertion-dependent order) -- so a baseline-diffing consumer (e.g. a
    code-audit baseline JSON) could see the SAME cycle reported under a DIFFERENT
    ``check::file:line`` key across scans of identical source, permanently flaking any "no new
    findings" gate built on it. ``scan_import_cycles`` now sorts ``comp`` before using
    ``comp[0]``, so the representative is the same regardless of traversal order."""
    from pyutilz.dev.code_audit.import_cycles import _strongly_connected_components

    edges = {"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": {"a"}}
    forward_order = dict(edges)
    reverse_order = {k: edges[k] for k in reversed(list(edges))}
    assert list(forward_order) != list(reverse_order), "test setup: insertion orders must actually differ"

    forward_scc = _strongly_connected_components(forward_order)
    reverse_scc = _strongly_connected_components(reverse_order)
    assert len(forward_scc) == 1 and len(reverse_scc) == 1
    assert set(forward_scc[0]) == set(reverse_scc[0]) == {"a", "b", "c", "d"}, "test setup: both must find the same 4-node cycle"

    # This is exactly what scan_import_cycles now does before using comp[0] (see the `comp =
    # sorted(comp)` line in scan_import_cycles) -- the representative must agree regardless of
    # which insertion order produced the SCC.
    assert sorted(forward_scc[0])[0] == sorted(reverse_scc[0])[0] == "a"


def test_threshold_below_documented_result_reads_a_comma_grouped_number_whole(tmp_path: Path):
    """A docstring saying "7,297 of 12,121" documents 7,297 - not 297.

    The leading word boundary matched only the LAST group, so the claim was read as 297 and every message
    quoted that. The reported number is what a reader checks the finding against, so a wrong one sends them
    looking for a gate that does not exist.

    This does NOT assert the finding disappears, and that is deliberate: the scanner pairs a documented count
    with any `>=` in the same function without knowing whether the two measure the same quantity, so a
    docstring about 7,297 priors beside an assertion about 2 classes per entry still reports. That is a
    separate limitation of the heuristic, and papering over it here by asserting an empty list would hide it.
    """
    _write(
        tmp_path,
        "test_grouped.py",
        '''
def test_most_priors_carry_no_basis():
    """7,297 of 12,121 stated priors carry no basis at all."""
    for entry in ([1, 2],):
        assert len(entry) >= 2
''',
    )
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings
    assert "documents 7297" in findings[0].detail, findings[0].detail
    assert "documents 297 " not in findings[0].detail, findings[0].detail


def test_threshold_below_documented_result_still_flags_a_grouped_claim_that_is_genuinely_weakened(tmp_path: Path):
    """The negative control: reading the number whole must not stop the check firing when it should."""
    _write(
        tmp_path,
        "test_grouped_bad.py",
        '''
def test_rows_recovered():
    """1,200 of 1,500 rows recover their source."""
    recovered = 1200
    assert recovered >= 900
''',
    )
    findings = scan_thresholds_below_documented_result(tmp_path)
    assert len(findings) == 1, findings


# ---- hardcoded_absolute_path_in_test -------------------------------------


def test_hardcoded_absolute_path_windows_drive_letter_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "D:\\\\Machine Learning\\\\data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "hardcoded_absolute_path_in_test"
    assert findings[0].severity == "P2"


def test_hardcoded_absolute_path_posix_home_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/home/alice/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_macos_users_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/Users/bob/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_root_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/root/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_ignores_non_test_file(tmp_path: Path):
    """The same literal in a non-test module is not flagged -- only test files are scanned."""
    _write(tmp_path, "helper.py", '''
def get_default_path():
    return "D:\\\\Machine Learning\\\\data.csv"
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_tmp_var_rooted_is_clean(tmp_path: Path):
    """A /tmp/-rooted or /var/-rooted literal is common/portable and NOT flagged."""
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/tmp/scratch/data.csv"
    assert p
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_tmp_path_fixture_is_clean(tmp_path: Path):
    """The correct pattern (tmp_path/Path(__file__).parent-derived) is never flagged."""
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture(tmp_path):
    p = tmp_path / "data.csv"
    assert p
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_detects_by_tests_directory(tmp_path: Path):
    """A file under a 'tests' directory is scanned even without a test_/​_test.py name."""
    sub = tmp_path / "tests"
    sub.mkdir()
    _write(sub, "fixtures.py", '''
DATA_PATH = "C:/Users/carol/fixture.csv"
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


# ---- async_primitive_reinit_per_call --------------------------------------


def test_async_primitive_reinit_lock_inside_function_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    lock = asyncio.Lock()
    async with lock:
        pass
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "async_primitive_reinit_per_call"
    assert findings[0].severity == "P1"


def test_async_primitive_reinit_semaphore_default_arg_flagged(tmp_path: Path):
    """A primitive constructed as a default-argument expression inside the function body is also flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    sem = asyncio.Semaphore(3)
    async with sem:
        pass
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1


def test_async_primitive_reinit_module_scope_is_clean(tmp_path: Path):
    """A primitive created at module scope (the correct pattern) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

_LOCK = asyncio.Lock()

async def handle():
    async with _LOCK:
        pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_init_attribute_is_clean(tmp_path: Path):
    """A primitive assigned to self in __init__ (created once per instance, shared across calls) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

class Worker:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def handle(self):
        async with self._lock:
            pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_global_lazy_singleton_is_clean(tmp_path: Path):
    """The global-declared lazy-module-singleton idiom is the safe shared-instance case."""
    _write(tmp_path, "mod.py", '''
import asyncio

_sem = None

async def get_sem():
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(5)
    return _sem
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_non_primitive_call_is_clean(tmp_path: Path):
    """An asyncio call that is NOT one of the coordination primitives (e.g. asyncio.sleep) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    await asyncio.sleep(0.1)
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_module_level_registry_is_clean(tmp_path: Path):
    """Publishing a primitive INTO a module-level container is the single-flight
    idiom -- every caller finds the same Event through the shared dict, which is
    the opposite of a private per-call copy. Needs no `global` (never rebound)."""
    _write(tmp_path, "mod.py", '''
import asyncio

_inflight: dict = {}
_inflight_lock = asyncio.Lock()

async def cached_get(key):
    async with _inflight_lock:
        if key not in _inflight:
            _inflight[key] = asyncio.Event()
            return None
        evt = _inflight[key]
    await evt.wait()
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_local_dict_registry_is_still_flagged(tmp_path: Path):
    """Guard on the exemption above: a FUNCTION-LOCAL dict is not shared, so
    publishing into it is still a private per-call copy and stays flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def cached_get(key):
    inflight = {}
    inflight[key] = asyncio.Event()
    await inflight[key].wait()
''')
    assert len(scan_async_primitive_reinit_per_call(tmp_path)) == 1


def test_async_primitive_reinit_bounded_gather_closure_is_clean(tmp_path: Path):
    """The bounded-gather idiom: the semaphore bounds the tasks THIS call
    spawns, so one per call is correct and deliberate."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def bounded_gather(factories, limit):
    sem = asyncio.Semaphore(limit)

    async def _run(factory):
        async with sem:
            return await factory()

    return await asyncio.gather(*[_run(f) for f in factories])
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_handed_to_helper_is_clean(tmp_path: Path):
    """Same idiom written with a helper instead of a closure."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def run_round(session, payload):
    sem = asyncio.Semaphore(4)
    return await _run_pipeline_for(session, sem, payload)
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_direct_use_still_flagged_alongside_a_closure(tmp_path: Path):
    """Non-vacuousness guard for BOTH fan-out exemptions: a function that also
    defines a closure must not become a blanket amnesty. The lock here is used
    directly in the body and never reaches the closure, so it stays flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle(items):
    lock = asyncio.Lock()

    async def _work(i):
        return i * 2

    async with lock:
        pass
    return await asyncio.gather(*[_work(i) for i in items])
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1, [f.detail for f in findings]


def test_async_primitive_reinit_custom_primitive_names(tmp_path: Path):
    """The primitive_names parameter can narrow/widen which asyncio.* constructors are tracked."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    lock = asyncio.Lock()
    async with lock:
        pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path, primitive_names=frozenset({"Event"})) == []


# ---- llm_call_missing_max_tokens_cap ---------------------------------------


def test_llm_max_tokens_cap_missing_kwarg_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello")
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "llm_call_missing_max_tokens_cap"
    assert findings[0].severity == "P2"


def test_llm_max_tokens_cap_zero_literal_flagged(tmp_path: Path):
    """An explicit max_tokens=0 is the same as omitting it -- still flagged."""
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello", max_tokens=0)
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 1


def test_llm_max_tokens_cap_generate_json_and_generate_batch_flagged(tmp_path: Path):
    """generate_json and generate_batch are also tracked capped methods."""
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate_json("hello")
provider.generate_batch(["a", "b"])
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 2


def test_llm_max_tokens_cap_explicit_nonzero_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello", max_tokens=2000)
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []


def test_llm_max_tokens_cap_non_provider_variable_is_clean(tmp_path: Path):
    """A .generate(...) call on a variable NOT assigned from get_llm_provider(...) is not tracked."""
    _write(tmp_path, "mod.py", '''
other = SomeUnrelatedClass()
other.generate("hello")
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []


def test_llm_max_tokens_cap_no_provider_in_module_short_circuits(tmp_path: Path):
    """A module with no get_llm_provider(...) assignment at all is skipped entirely (cheap early-out)."""
    _write(tmp_path, "mod.py", '''
def f():
    return 1
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []


# ---- __main__ module-execution guard ---------------------------------------


def test_dunder_main_module_execution_delegates_to_cli_main(tmp_path: Path):
    """``python -m pyutilz.dev.code_audit <root>`` runs the ``if __name__ == '__main__'`` guard in
    __main__.py, which only executes under real module execution (never when pytest imports the
    module normally) -- exercised here via a real subprocess."""
    import subprocess
    import sys

    _write(tmp_path, "ok.py", "def f(x=None):\n    return x\n")
    result = subprocess.run(  # nosec B603 -- fixed local argv, no shell, no untrusted input
        [sys.executable, "-m", "pyutilz.dev.code_audit", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


# ---- exclude_dirs must judge the path BELOW the root, never the root's own ancestors --------


def test_an_excluded_name_ABOVE_the_scan_root_does_not_silence_the_scan(tmp_path: Path):
    """A scan root living inside a directory named in ``exclude_dirs`` must still be scanned.

    The check used to match ``exclude_dirs`` against the ABSOLUTE path's components, so every file
    under a checkout at ``.../.claude/worktrees/<agent>/`` matched on the ancestor ``.claude`` and
    the scan returned nothing at all - for a whole package, silently, with every audit built on it
    passing vacuously. That is where Claude Code agent worktrees live, and the same trap sits one
    directory away for any project under ``build/``, ``dist/``, ``env/`` or ``venv/``.
    """
    buried = tmp_path / ".claude" / "worktrees" / "agent-1" / "pkg"
    buried.mkdir(parents=True)
    (buried / "mod.py").write_text("def obviously_dead_helper(x):\n    return x\n", encoding="utf-8")

    findings = scan_dead_public_callables(buried)
    assert [f.detail for f in findings if "obviously_dead_helper" in f.detail], "a root under a `.claude` ancestor was scanned as if empty"


def test_an_excluded_name_BELOW_the_scan_root_is_still_skipped(tmp_path: Path):
    """The negative control: the exclusion itself must keep working for real build/cache dirs."""
    inner = tmp_path / "__pycache__"
    inner.mkdir()
    (inner / "mod.py").write_text("def obviously_dead_helper(x):\n    return x\n", encoding="utf-8")

    assert not [f for f in scan_dead_public_callables(tmp_path) if "obviously_dead_helper" in f.detail]


def test_every_tree_walking_scanner_agrees_between_a_relative_and_absolute_root(tmp_path: Path):
    """A scan must not depend on how its root was spelled.

    `_iter_py_files` was fixed first, and two scanners turned out to carry their OWN copy of the
    exclude check against the absolute path - so the same tree scanned as `Path("tests")` and as
    `Path("tests").resolve()` gave different answers whenever the checkout sat under an excluded
    ancestor. Both are routed through `_is_excluded` now; this pins that they cannot drift apart again.
    """
    import os

    from pyutilz.dev.code_audit import scan_import_cycles, scan_redundant_test_fit_calls

    pkg = tmp_path / ".claude" / "worktrees" / "agent-1" / "proj"
    (pkg / "tests").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tests" / "test_thing.py").write_text(
        "from functools import lru_cache\n\n\n"
        "@lru_cache\ndef fit(a):\n    return a\n\n\n"
        "def test_one():\n    assert fit(1) == 1\n\n\n"
        "def test_two():\n    assert fit(1) == 1\n",
        encoding="utf-8",
    )

    cwd = os.getcwd()
    try:
        os.chdir(pkg)
        for scan, args in ((scan_redundant_test_fit_calls, (Path("tests"),)), (scan_import_cycles, (Path("."), "proj"))):
            relative = len(list(scan(*args)))
            absolute = len(list(scan(args[0].resolve(), *args[1:])))
            assert relative == absolute, f"{scan.__name__} saw {relative} findings via a relative root and {absolute} via an absolute one"
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# check_all / normalise_text / get_scanners -- all three are in dev.code_audit's __all__
# and were previously never mentioned anywhere under tests/ (audit F20, 2026-09-02).
# get_scanners and the registry it copies are part of the shared meta-test harness pyutilz
# exports to its downstream consumers, so a regression here breaks six other repos.
# ---------------------------------------------------------------------------


def test_normalise_text_deletes_intra_word_hyphens_so_both_spellings_are_one_token():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("Ante-Mortem") == normalise_text("antemortem") == "antemortem"
    assert normalise_text("POST-mortem") == "postmortem"


def test_normalise_text_leaves_edge_and_digit_adjacent_hyphens_alone():
    """Only a hyphen BETWEEN two letters joins a compound; a leading/trailing one or a
    letter-digit boundary is meaningful punctuation, not a spelling variant."""
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("co-2") == "co-2"
    assert normalise_text("-lead") == "-lead"
    assert normalise_text("trail-") == "trail-"


def test_normalise_text_collapses_whitespace_and_underscores():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text("  Multiple   spaces\tand\nnewlines  ") == "multiple spaces and newlines"
    assert normalise_text("snake_case_name") == "snake case name"


def test_normalise_text_handles_none_and_non_strings():
    from pyutilz.dev.code_audit.field_text_agreement import normalise_text

    assert normalise_text(None) == ""
    assert normalise_text("") == ""
    assert normalise_text(42) == "42"


def test_check_all_returns_one_report_per_rule_keyed_by_name():
    from pyutilz.dev.code_audit import FieldTextRule, check_all

    rows = [
        {"subject": "haemorrhage", "object": "vital hanging", "temporal_class": "antemortem"},
        {"subject": "", "object": "putrefaction", "temporal_class": "postmortem"},
    ]
    manner = FieldTextRule(name="manner", field="manner", text_fields=("object",))
    reports = check_all([_temporal_rule(), manner], rows)

    assert set(reports) == {"temporal_class", "manner"}
    assert reports["temporal_class"].agree == 2
    assert reports["temporal_class"].contradict == 0
    assert reports["temporal_class"].n_records == 2


def test_check_all_includes_vocabulary_less_rules_instead_of_dropping_them():
    """A rule with no cues must appear as 100% uncheckable, not vanish from the output -- an
    unmeasured pair silently missing from a report reads as a pair that passed."""
    from pyutilz.dev.code_audit import FieldTextRule, check_all

    rows = [{"object": "anything", "manner": "blunt"}]
    reports = check_all([FieldTextRule(name="manner", field="manner", text_fields=("object",))], rows)

    assert "manner" in reports
    assert reports["manner"].has_vocabulary is False
    assert reports["manner"].uncheckable == 1
    assert reports["manner"].coverage == 0.0


def test_check_all_accepts_explicit_record_ids():
    from pyutilz.dev.code_audit import check_all

    rows = [{"subject": "", "object": "vital hanging", "temporal_class": "postmortem"}]
    reports = check_all([_temporal_rule()], rows, ids=["record-7"])

    assert reports["temporal_class"].contradict == 1
    assert reports["temporal_class"].contradictions[0].record_id == "record-7"


def test_get_scanners_returns_a_populated_registry_of_callables():
    from pyutilz.dev.code_audit import get_scanners

    scanners = get_scanners()
    assert isinstance(scanners, dict)
    assert scanners, "the scanner registry must not be empty"
    assert all(callable(fn) for fn in scanners.values())
    # A few names every consumer's baseline depends on.
    for expected in ("vacuous_assertion", "except_skip_masks_call_under_test", "bare_except"):
        assert expected in scanners, f"{expected!r} missing from the registry: {sorted(scanners)}"


def test_get_scanners_returns_a_copy_so_callers_cannot_corrupt_the_shared_registry():
    """The documented reason this accessor exists at all: ``get_scanners().pop(...)`` must not
    silently disarm a scanner for every subsequent run_all() in the same process."""
    from pyutilz.dev.code_audit import get_scanners

    first = get_scanners()
    victim = next(iter(first))
    first.pop(victim)
    first["definitely_not_a_real_scanner"] = lambda *a, **kw: []

    second = get_scanners()
    assert victim in second
    assert "definitely_not_a_real_scanner" not in second
# ---- source_text_assertion ----------------------------------------------
#
# The defect this scanner exists for has shipped twice: a test asserted a fix was present in a
# function's SOURCE, the source did contain it, and the function was never reached. Every case
# below is written from a real spelling seen in the audited repos rather than from the shape the
# scanner happens to implement -- the first version of the scanner matched only the inline form
# and reported zero offences in a repo full of them.


def test_source_text_assertion_flags_read_into_a_variable(tmp_path: Path):
    _write(
        tmp_path,
        "test_thing.py",
        """
import inspect
import mod

def test_the_fix_landed():
    src = inspect.getsource(mod.handler)
    assert "AT TIME ZONE 'utc'" in src
""",
    )
    findings = scan_source_text_assertions(tmp_path)
    assert len(findings) == 1
    assert findings[0].line == 6
    assert "getsource" in findings[0].detail


def test_source_text_assertion_flags_the_inline_form(tmp_path: Path):
    _write(
        tmp_path,
        "test_thing.py",
        """
import inspect
import mod

def test_it():
    assert "retries=3" in inspect.getsource(mod.fetch)
""",
    )
    assert len(scan_source_text_assertions(tmp_path)) == 1


def test_source_text_assertion_flags_reading_a_sql_file(tmp_path: Path):
    _write(
        tmp_path,
        "test_thing.py",
        """
from pathlib import Path

def test_the_index_exists():
    sql = Path("sql/schema.sql").read_text()
    assert "CREATE INDEX ix_jobs_ts" in sql
""",
    )
    findings = scan_source_text_assertions(tmp_path)
    assert len(findings) == 1
    assert ".sql" in findings[0].detail


def test_source_text_assertion_ignores_a_behavioural_assertion(tmp_path: Path):
    """The honest version of the same test: call the code, assert on what comes back."""
    _write(
        tmp_path,
        "test_thing.py",
        """
import mod

def test_the_fix_landed():
    assert "AT TIME ZONE 'utc'" in mod.build_query()
""",
    )
    assert scan_source_text_assertions(tmp_path) == []


def test_source_text_assertion_ignores_calling_an_unwrapped_callable(tmp_path: Path):
    """Reaching through `__code__` to pull a decorated function out of its closure, then CALLING
    it, is behavioural testing -- an earlier version of the scanner mislabelled it."""
    _write(
        tmp_path,
        "test_thing.py",
        """
def test_a_real_tab_switch_still_rebuilds(app):
    wrapped = app.callback_map["k"]["callback"]
    fn = wrapped.__closure__[wrapped.__code__.co_freevars.index("func")].cell_contents
    assert fn("tabMarket") == "body:tabMarket"
""",
    )
    assert scan_source_text_assertions(tmp_path) == []


def test_source_text_assertion_ignores_reading_source_without_claiming_content(tmp_path: Path):
    _write(
        tmp_path,
        "test_thing.py",
        """
import inspect
import mod

def test_it_is_introspectable():
    assert inspect.getsource(mod.handler)
""",
    )
    assert scan_source_text_assertions(tmp_path) == []


def test_source_text_assertion_ignores_non_test_files(tmp_path: Path):
    """A code generator or build script manipulates source text as its actual job."""
    _write(
        tmp_path,
        "codegen.py",
        """
import inspect
import mod

def check():
    src = inspect.getsource(mod.handler)
    assert "def handler" in src
""",
    )
    assert scan_source_text_assertions(tmp_path) == []


def test_source_text_assertion_scopes_bound_names_per_function(tmp_path: Path):
    """`src` is an ordinary local name that recurs across a file. Binding it file-wide made an
    unrelated behavioural assertion in a later test look like a source-text claim."""
    _write(
        tmp_path,
        "test_thing.py",
        """
import inspect
import mod

def test_one():
    src = inspect.getsource(mod.handler)
    assert "marker" in src

def test_two():
    src = mod.render_template()
    assert "marker" in src
""",
    )
    findings = scan_source_text_assertions(tmp_path)
    assert len(findings) == 1, [f.line for f in findings]
    assert findings[0].line == 6


# ---- docstring_numbers_moved_to_config (opt-in) --------------------------
#
# Opt-in, so these tests are the only place it is exercised by default. Its precision was measured
# rather than assumed: three hits across four repos, all false, which is why it is not in a ratchet.
# The negative cases below are those three, kept as tests so a future widening of the rule cannot
# quietly reintroduce them.


def test_docstring_numbers_moved_to_config_flags_the_stale_prose(tmp_path: Path):
    _write(
        tmp_path,
        "prune.py",
        """
def _prune_disappearance_counts(state):
    \"\"\"Drop sources that have disappeared.

    Prunes a source after 10 consecutive misses, or 5 for a rare source.
    \"\"\"
    from live_config import cfg
    common = cfg().get("prune", "common_misses", None, int)
    rare = cfg().get("prune", "rare_misses", None, int)
    return {k: v for k, v in state.items() if v < (rare if k in state else common)}
""",
    )
    findings = scan_docstring_numbers_moved_to_config(tmp_path)
    assert len(findings) == 1
    assert "10" in findings[0].detail and "5" in findings[0].detail


def test_docstring_numbers_moved_to_config_ignores_a_named_source(tmp_path: Path):
    """Naming the constant is the RECOMMENDED form; flagging it would punish the fix."""
    _write(
        tmp_path,
        "resolve.py",
        """
def resolve_min_days(cli=None):
    \"\"\"Resolve the rescan floor in days.

    Precedence: CLI > config key > compiled default (``MIN_WH_RESCAN_FREQ_DAYS`` = 14).
    \"\"\"
    from live_config import cfg
    return int(cfg().get("intervals", "min_days", MIN_WH_RESCAN_FREQ_DAYS, int))
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_a_document_reference(tmp_path: Path):
    """ "audit 04.1" is a citation, not a threshold -- it got through because the same line said
    "after"."""
    _write(
        tmp_path,
        "banners.py",
        """
def _mode_banners(settings):
    \"\"\"Banners built from a fresh settings read.

    The confirmation modal (audit 04.1) closes the window after every tick.
    \"\"\"
    from live_config import cfg
    return cfg().get("submission", "dry_run", None, bool)
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_numbers_still_in_the_body(tmp_path: Path):
    """If the number is in the code, the prose can be checked against it by reading."""
    _write(
        tmp_path,
        "prune.py",
        """
def prune(state):
    \"\"\"Prunes a source after 10 consecutive misses.\"\"\"
    from live_config import cfg
    limit = cfg().get("prune", "misses", 10, int)
    return {k: v for k, v in state.items() if v < limit}
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_ignores_a_function_reading_no_config(tmp_path: Path):
    _write(
        tmp_path,
        "prune.py",
        """
def describe():
    \"\"\"The batch size limit is 500 per call.\"\"\"
    return LIMIT
""",
    )
    assert scan_docstring_numbers_moved_to_config(tmp_path) == []


def test_docstring_numbers_moved_to_config_is_opt_in():
    """It must not reach any project's default run, and therefore any project's baseline."""
    from pyutilz.dev.code_audit import OPT_IN_ONLY, get_scanners

    assert "docstring_numbers_moved_to_config" in get_scanners()
    assert "docstring_numbers_moved_to_config" in OPT_IN_ONLY


# ---- raising_stub_swallowed ---------------------------------------------
#
# A test says "this must never be called" by raising, and a broad handler downstream turns the
# raise into a benign path. Confirmed: a cache was re-probed on every run behind a green test.


def test_raising_stub_swallowed_flags_the_shape(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def part_has_upwork(part):
    try:
        return ParquetFile(part).probe()
    except Exception:
        return None
""")
    _write(tmp_path, "test_cache.py", """
from unittest.mock import patch

def test_locate_uses_cache_when_not_stale():
    def _boom(*a, **k):
        raise AssertionError("must not be called")
    with patch("prod.ParquetFile", _boom) as spy:
        locate()
    assert spy.called is False
""")
    findings = scan_raising_stub_swallowed(tmp_path)
    assert len(findings) == 1
    assert "ParquetFile" in findings[0].detail


def test_raising_stub_swallowed_scopes_stub_names_to_the_test(tmp_path: Path):
    """`_gql` is a name every test in a file defines for itself. Collected module-wide, one
    raising definition tainted five harmless ones and produced the rule's only false positive on a
    real repository."""
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except Exception:
        return None
""")
    _write(tmp_path, "test_x.py", """
from unittest.mock import patch

def test_harmless():
    def _gql(*a, **k):
        return {"ok": True}
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called

def test_raising():
    def _gql(*a, **k):
        raise AssertionError("no")
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called is False
""")
    findings = scan_raising_stub_swallowed(tmp_path)
    assert len(findings) == 1, [f.line for f in findings]


def test_raising_stub_swallowed_ignores_a_test_expecting_the_raise(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except Exception:
        return None
""")
    _write(tmp_path, "test_x.py", """
import pytest
from unittest.mock import patch

def test_it_propagates():
    def _gql(*a, **k):
        raise ValueError("boom")
    with patch("prod.gql", _gql) as spy:
        with pytest.raises(ValueError):
            go()
    assert spy.called
""")
    assert scan_raising_stub_swallowed(tmp_path) == []


def test_raising_stub_swallowed_ignores_a_narrow_handler(tmp_path: Path):
    """A handler that catches a specific type is not the swallow this rule is about."""
    _write(tmp_path, "prod.py", """
def go():
    try:
        return gql(1)
    except KeyError:
        return None
""")
    _write(tmp_path, "test_x.py", """
from unittest.mock import patch

def test_x():
    def _gql(*a, **k):
        raise AssertionError("no")
    with patch("prod.gql", _gql) as spy:
        go()
    assert spy.called is False
""")
    assert scan_raising_stub_swallowed(tmp_path) == []


# ---- lazy_log_assertion --------------------------------------------------


def test_lazy_log_assertion_flags_a_formatted_expectation(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go(label, r, t):
    log.warning("%s: reached only %d/%d", label, r, t)
""")
    _write(tmp_path, "test_x.py", """
def test_shortfall_is_warned():
    assert "reached only 0/3" in str(log.warning.call_args)
""")
    findings = scan_lazy_log_assertion(tmp_path)
    assert len(findings) == 1
    assert "reached only 0/3" in findings[0].detail


def test_lazy_log_assertion_ignores_an_fstring_rendering(tmp_path: Path):
    """f-strings format EAGERLY, so the values DO reach the record and the assertion can match.
    Both of this rule's first hits on a real repository were this."""
    _write(tmp_path, "prod.py", """
def go(n):
    log.info(f"Found {n} on-disk checkpoint(s) -- these will resume")
""")
    _write(tmp_path, "test_x.py", """
def test_inventory_is_logged():
    assert any("Found 2 on-disk checkpoint" in str(c) for c in log.info.call_args_list)
""")
    assert scan_lazy_log_assertion(tmp_path) == []


def test_lazy_log_assertion_ignores_a_bare_value(tmp_path: Path):
    """`"j1"` is an id the test supplied, with no message text around it -- production logs it
    through an f-string, so it really is in args[0]."""
    _write(tmp_path, "prod.py", """
def go(jid, e):
    log.warning(f"Reconcile sample failed for {jid}: {e}")
""")
    _write(tmp_path, "test_x.py", """
def test_error_logged():
    assert "j1" in log.warning.call_args[0][0]
""")
    assert scan_lazy_log_assertion(tmp_path) == []


def test_lazy_log_assertion_ignores_a_format_that_carries_its_own_digit(tmp_path: Path):
    _write(tmp_path, "prod.py", """
def go(host):
    log.warning("HTTP 429 from %s", host)
""")
    _write(tmp_path, "test_x.py", """
def test_rate_limit_logged():
    assert "HTTP 429 from" in str(log.warning.call_args)
""")
    assert scan_lazy_log_assertion(tmp_path) == []


# ---- constructor_param_overwritten ---------------------------------------


def test_constructor_param_overwritten_follows_one_call_hop(tmp_path: Path):
    """The worked example assigns through a second method: `_refresh_rate` reads config and calls
    `update_rate(rate)`, which does the assignment. Requiring both in one statement missed it."""
    _write(tmp_path, "bucket.py", """
class TokenBucket:
    def __init__(self, rate):
        self._rate = rate

    def update_rate(self, rate):
        self._rate = rate

    def _refresh_rate(self):
        self.update_rate(cfg().get("traffic", "max_rps", 10.0, float))
""")
    findings = scan_constructor_param_overwritten(tmp_path)
    assert len(findings) == 1
    assert "_refresh_rate" in findings[0].detail
    assert "update_rate" in findings[0].detail


def test_constructor_param_overwritten_ignores_a_stable_attribute(tmp_path: Path):
    _write(tmp_path, "bucket.py", """
class Plain:
    def __init__(self, rate):
        self._rate = rate

    def use(self):
        return self._rate * 2
""")
    assert scan_constructor_param_overwritten(tmp_path) == []


def test_constructor_param_overwritten_ignores_a_reassignment_not_from_config(tmp_path: Path):
    """Reassigning from an argument is ordinary mutation, not the deployment overriding a test."""
    _write(tmp_path, "bucket.py", """
class Plain:
    def __init__(self, rate):
        self._rate = rate

    def set_rate(self, rate):
        self._rate = rate
""")
    assert scan_constructor_param_overwritten(tmp_path) == []


# ---- stats_key_coverage --------------------------------------------------
#
# The audited crawler recorded two incidents on one dict: a lazily-created counter that was
# cumulative since process start while every sibling was per-cycle, and an unregistered key that
# turned an increment helper into a KeyError. A third happened while this rule was being written.


def test_stats_key_coverage_flags_an_undeclared_accumulating_key(tmp_path: Path):
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0, "dups": 0}

    def note(self, n):
        self.stats["skipped_small"] = self.stats.get("skipped_small", 0) + n
""",
    )
    findings = scan_stats_key_coverage(tmp_path)
    assert len(findings) == 1
    assert "skipped_small" in findings[0].detail


def test_stats_key_coverage_matches_across_mixins(tmp_path: Path):
    """The real shape: one class declares the dict, another increments it, and they are one
    object only at runtime. Scoped per class, this rule missed the bug it was written from."""
    _write(
        tmp_path,
        "stats_mixin.py",
        """
class StatsMixin:
    def _reset_stats(self):
        self.stats = {"pages": 0}
""",
    )
    _write(
        tmp_path,
        "split_mixin.py",
        """
class SplitMixin:
    def split(self):
        self._inc_stat("overlapping_axis_skipped")
""",
    )
    findings = scan_stats_key_coverage(tmp_path)
    assert len(findings) == 1
    assert "overlapping_axis_skipped" in findings[0].detail


def test_stats_key_coverage_ignores_a_plain_assignment(tmp_path: Path):
    """`self.stats["k"] = value` overwrites completely every cycle, so it is safe undeclared.
    Four such keys in the audited crawler were this rule's only false positives."""
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0}

    def finish(self, recovered):
        self.stats["recovered"] = recovered
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


def test_stats_key_coverage_ignores_a_declared_key(tmp_path: Path):
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0, "skipped_small": 0}

    def note(self, n):
        self.stats["skipped_small"] += n
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


def test_stats_key_coverage_ignores_a_class_that_declares_nothing(tmp_path: Path):
    """A dict with no declared shape has no contract to violate."""
    _write(
        tmp_path,
        "loose.py",
        """
class Loose:
    def note(self, n):
        self.counts["whatever"] = self.counts.get("whatever", 0) + n
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


# ---- sentinel_guard_mismatch ---------------------------------------------
#
# A failure path returns a falsy value while the caller guards `is None`, so the failure reads as
# a legitimate answer. One transient HTML error page retired a discovery source permanently.


def test_sentinel_guard_mismatch_flags_the_canonical_case(tmp_path: Path):
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return 0
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total is None or (pages_total > 0 and state.done):
        return
    mark_source_empty(state)
""",
    )
    findings = scan_sentinel_guard_mismatch(tmp_path)
    assert len(findings) == 1
    assert "fetch_num_pages" in findings[0].detail


def test_sentinel_guard_mismatch_allows_none_as_a_third_answer(tmp_path: Path):
    """The accepted fix for this shape was, verbatim, to make None a third answer. A function that
    returns None for the failure and a falsy value for a real outcome is the FIXED form."""
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return None
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total is None:
        return
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


def test_sentinel_guard_mismatch_ignores_a_falsy_return_on_the_ordinary_path(tmp_path: Path):
    """A function returning 0 from its normal path is returning a number, not signalling."""
    _write(
        tmp_path,
        "counts.py",
        """
def how_many(items):
    if not items:
        return 0
    return len(items)
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(items):
    n = how_many(items)
    if n is None:
        return
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


def test_sentinel_guard_mismatch_needs_a_caller_that_guards_on_none(tmp_path: Path):
    """Returning 0 on failure is fine if nobody tests the result for None."""
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return 0
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total > 0:
        go(pages_total)
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


# ---- unit_suffix_mismatch ------------------------------------------------
#
# A quantity stored under one unit and read from another. A `duration_s` column measured cycle
# wall-clock while the real work time sat one JSONB level away as `extra.minutes`.


def test_unit_suffix_mismatch_flags_a_bare_cross_unit_read(tmp_path: Path):
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    work_s = totals["minutes"]
    return work_s
""",
    )
    findings = scan_unit_suffix_mismatch(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_unit_suffix_mismatch_is_silent_on_a_conversion(tmp_path: Path):
    """`work_s = totals["minutes"] * 60` is the CORRECT form. Any arithmetic counts as a
    conversion -- assuming otherwise would flag every correct conversion in a tree."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    work_s = totals["minutes"] * 60
    return work_s
""",
    )
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_treats_synonyms_as_one_unit(tmp_path: Path):
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    elapsed_secs = totals["seconds"]
    return elapsed_secs
""",
    )
    assert scan_unit_suffix_mismatch(tmp_path) == []


def test_unit_suffix_mismatch_covers_keyword_arguments(tmp_path: Path):
    """The audited case passed the wrong unit as a keyword to the recorder, not via assignment."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(extra):
    record_run(duration_s=extra["minutes"])
""",
    )
    assert len(scan_unit_suffix_mismatch(tmp_path)) == 1


def test_unit_suffix_mismatch_ranks_a_cross_family_pair_lower(tmp_path: Path):
    """Seconds against bytes is more likely a naming coincidence than a real conversion bug."""
    _write(
        tmp_path,
        "obs.py",
        """
def record(totals):
    payload_bytes = totals["seconds"]
    return payload_bytes
""",
    )
    findings = scan_unit_suffix_mismatch(tmp_path)
    assert len(findings) == 1 and findings[0].severity == "Low"


def test_unit_suffix_mismatch_reads_the_pre_3_9_subscript_shape():
    """Up to python 3.8 the parser wrapped `d["key"]` in an `ast.Index` node, which 3.9 removed
    (bpo-34822). Reading `.slice` without unwrapping matched nothing on 3.8, so the whole rule went
    quiet there rather than erroring. Simulated with a legacy-shaped node, since the interpreter
    running this test builds the modern shape."""
    import ast as _ast

    from pyutilz.dev.code_audit._base import _subscript_index
    from pyutilz.dev.code_audit.unit_suffix_mismatch import _source_name, _target_names

    class Index(_ast.AST):  # the node class python<=3.8 actually produced, named exactly as it was
        _fields = ("value",)

    # Built node-by-node rather than parsed: python 3.8 ALREADY produces this shape, so wrapping a
    # parsed subscript there would nest Index inside Index and prove nothing.
    const = _ast.Constant(value="minutes")
    legacy = _ast.Subscript(value=_ast.Name(id="totals", ctx=_ast.Load()), slice=Index(value=const), ctx=_ast.Load())

    assert isinstance(_subscript_index(legacy), _ast.Constant)
    assert _source_name(legacy) == "minutes"
    assert _target_names(legacy) == ["minutes"]


# ---- comment_names_missing_symbol ----------------------------------------
#
# Prose that points somewhere is trusted. One such comment WAS the accepted mitigation for an
# earlier SQL-injection finding and named a helper that had since been renamed.


def test_comment_names_missing_symbol_flags_a_rotted_private_pointer(tmp_path: Path):
    _write(
        tmp_path,
        "perm.py",
        """
# The SQL is built by `_perm_err_sql()`, which escapes every pattern.
def perm_err_text_like_sql(patterns):
    return " OR ".join(patterns)
""",
    )
    findings = scan_comment_names_missing_symbol(tmp_path)
    assert len(findings) == 1
    assert "_perm_err_sql" in findings[0].detail


def test_comment_names_missing_symbol_ignores_library_methods(tmp_path: Path):
    """Unrestricted, this rule gave 52 hits in one package with no rotted pointer among them:
    `close()`, `min()`, `utcnow()`, `is_nan()`, `to_plotly_json()`, `model_dump()`. A leading
    underscore is the only reliable "this must be local" signal."""
    _write(
        tmp_path,
        "frames.py",
        """
# Values are dropped with `dropna()` and checked with `is_nan()` before `to_numpy()`.
def clean(df):
    return df
""",
    )
    assert scan_comment_names_missing_symbol(tmp_path) == []


def test_comment_names_missing_symbol_resolves_across_the_tree(tmp_path: Path):
    """A comment may cite a private helper defined in another module."""
    _write(
        tmp_path,
        "helpers.py",
        """
def _capped(n):
    return min(n, 100)
""",
    )
    _write(
        tmp_path,
        "user.py",
        """
# Capped by `_capped()` before use.
def go(n):
    return n
""",
    )
    assert scan_comment_names_missing_symbol(tmp_path) == []


def test_comment_cites_absolute_line_is_opt_in():
    """225 hits in one package, most of them legitimate coverage annotations. It reports rather
    than gates, so it cannot reach a project's default run or its baseline."""
    from pyutilz.dev.code_audit import OPT_IN_ONLY, get_scanners

    assert "comment_cites_absolute_line" in get_scanners()
    assert "comment_cites_absolute_line" in OPT_IN_ONLY


def test_comment_cites_absolute_line_finds_a_citation(tmp_path: Path):
    _write(
        tmp_path,
        "mod.py",
        """
# The unlink happens at line 619, after the flush.
def go():
    pass
""",
    )
    findings = scan_comment_cites_absolute_line(tmp_path)
    assert len(findings) == 1 and "619" in findings[0].detail


# ---- unreachable_import_fallback -----------------------------------------


def test_unreachable_import_fallback_flags_a_dead_guard(tmp_path: Path):
    """The handler cannot run, and its comment advertises a degradation path that never has."""
    _write(
        tmp_path,
        "mod.py",
        """
import struct

def parse(b):
    try:
        import struct
    except ImportError:
        return None
    return struct.unpack("<I", b)
""",
    )
    findings = scan_unreachable_import_fallback(tmp_path)
    assert len(findings) == 1
    assert "struct" in findings[0].detail


def test_unreachable_import_fallback_allows_an_optional_submodule(tmp_path: Path):
    """`import pkg.optional` can fail on a missing dependency where `import pkg` cannot. Comparing
    only the top-level package reported thirteen honest optional-dependency guards as dead."""
    _write(
        tmp_path,
        "mod.py",
        """
import pkg

def go():
    try:
        import pkg.optional
    except ImportError:
        return None
    return pkg.optional
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_a_genuinely_optional_import(tmp_path: Path):
    _write(
        tmp_path,
        "mod.py",
        """
def go():
    try:
        import orjson
    except ImportError:
        import json as orjson
    return orjson
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_the_type_checking_idiom(tmp_path: Path):
    """A TYPE_CHECKING-only import is not an import at runtime, so a try/except
    ImportError beside it is reachable, not dead.

    This is the standard shape for an optional dependency: import it under
    TYPE_CHECKING so a function can carry a real return annotation, and guard the
    real import so the package staying absent is handled. Flagging it told the
    author to delete a handler their code demonstrably needs.
    """
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None
''')
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_allows_the_qualified_type_checking_form(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
import typing

if typing.TYPE_CHECKING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert scan_unreachable_import_fallback(tmp_path) == []


def test_unreachable_import_fallback_still_flags_a_real_dead_guard_alongside_type_checking(tmp_path: Path):
    """Non-vacuousness. The exemption is for imports INSIDE the TYPE_CHECKING
    block; a genuinely unconditional runtime import elsewhere in the same file
    still makes the handler dead, and a file that happens to use TYPE_CHECKING
    must not become exempt wholesale."""
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections

import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert len(scan_unreachable_import_fallback(tmp_path)) == 1


def test_unreachable_import_fallback_does_not_exempt_a_compound_condition(tmp_path: Path):
    """``if TYPE_CHECKING or X:`` can execute at runtime, so treating it as
    type-only would hide the dead handler this rule exists to find."""
    _write(tmp_path, "mod.py", '''
from typing import TYPE_CHECKING

SOMETHING = True

if TYPE_CHECKING or SOMETHING:
    import tiktoken

try:
    import tiktoken
except ImportError:
    tiktoken = None
''')
    assert len(scan_unreachable_import_fallback(tmp_path)) == 1


def test_unreachable_import_fallback_ignores_an_import_guarded_everywhere(tmp_path: Path):
    """If EVERY import of the module is itself inside a try, none of them is certain."""
    _write(
        tmp_path,
        "mod.py",
        """
try:
    import cupy
except ImportError:
    cupy = None

def go():
    try:
        import cupy
    except ImportError:
        return None
    return cupy
""",
    )
    assert scan_unreachable_import_fallback(tmp_path) == []


# ---- asymmetric_except_siblings ------------------------------------------
#
# Run against a real repository this found `SafeDB.rollback` calling `_reconnect()` bare while
# `_retry_resource_error` wrapped the identical call -- and 21 of that package's 51 rollback call
# sites are inside an `except`, so a failing reconnect aborted whatever was recovering.


def test_asymmetric_except_siblings_flags_the_unguarded_twin(tmp_path: Path):
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def already_in_db(self, cid):
        try:
            return self.db.query(cid)
        except psycopg2.Error:
            self.db.rollback()
            return False

    def already_in_db_batch(self, cids):
        try:
            return self.db.query_many(cids)
        except psycopg2.Error:
            try:
                self.db.rollback()
            except Exception:
                pass
            return []
""",
    )
    findings = scan_asymmetric_except_siblings(tmp_path)
    assert len(findings) == 1
    assert "already_in_db" in findings[0].detail
    assert "rollback" in findings[0].detail


def test_asymmetric_except_siblings_needs_the_same_exception_type(tmp_path: Path):
    """Two handlers doing genuinely different jobs share neither the type nor the call, and
    comparing them would report every class with two try blocks in it."""
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except psycopg2.Error:
            self.db.rollback()

    def b(self):
        try:
            go()
        except OSError:
            try:
                self.db.rollback()
            except Exception:
                pass
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []


def test_asymmetric_except_siblings_is_silent_when_both_guard(tmp_path: Path):
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass

    def b(self):
        try:
            go()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []


def test_asymmetric_except_siblings_needs_two_siblings(tmp_path: Path):
    """A lone unguarded handler is a judgement call, not drift. The rule reports only where the
    same class already does it the other way."""
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except Exception:
            self.db.rollback()
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []


# ---- effect_flag_outside_its_effect --------------------------------------
#
# A success record set beside, rather than inside, the conditional work it records. An empty crawl
# advertised a parquet file it had never written.


def test_effect_flag_outside_its_effect_flags_the_canonical_case(tmp_path: Path):
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(table, _kind_ok, path):
    if table.num_rows:
        write_parquet(table, path, "pq")
    _kind_ok["pq"] = True
""",
    )
    findings = scan_effect_flag_outside_its_effect(tmp_path)
    assert len(findings) == 1
    assert "pq" in findings[0].detail


def test_effect_flag_outside_its_effect_accepts_the_record_inside_the_block(tmp_path: Path):
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(table, _kind_ok, path):
    if table.num_rows:
        write_parquet(table, path, "pq")
        _kind_ok["pq"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_understands_an_early_exit(tmp_path: Path):
    """A failure branch ending in `continue` puts the record on the other path by construction.
    That is the shape the audited codebase adopted when it FIXED this defect, and without modelling
    it the rule reports the fix as the bug -- which it did, twice, on real code."""
    _write(
        tmp_path,
        "out.py",
        """
def write_kinds(crawls, _kind_ok):
    for crawl in crawls:
        if not wrote(crawl, "pq"):
            log("skipped")
            continue
        _kind_ok["pq"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_requires_a_shared_name(tmp_path: Path):
    """Without the shared token this would flag every assignment that follows an `if`."""
    _write(
        tmp_path,
        "out.py",
        """
def go(table, flags, path):
    if table.num_rows:
        write_parquet(table, path)
    flags["something_else"] = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_self_only_link(tmp_path: Path):
    """`self` is mentioned by nearly every statement in a method, so it links nothing.

    Found against the repo itself: `self._ready = True` after an unrelated
    `if self._process.stdout:` was reported purely because both mention `self`.
    """
    _write(
        tmp_path,
        "out.py",
        """
class C:
    def start(self):
        if self.stdout:
            self.stream = wrap(self.stdout)
        self.ready = True
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_logging_only_guard(tmp_path: Path):
    """A guard whose body only logs guards no work, so the statement after it is not a record.

    Found against the repo itself: `if verbose: logger.info(...)` followed by the unconditional
    `res.add(str(obj))` -- reporting that inverts the rule.
    """
    _write(
        tmp_path,
        "out.py",
        """
def go(obj, res, verbose):
    if verbose:
        logger.info("Processing %s of size %s", type(obj), len(str(obj)))
    res.add(str(obj))
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_a_lazy_init_guard(tmp_path: Path):
    """``if key not in d: d[key] = set()`` creates an empty container; it is not
    the work a following ``d[key].add(...)`` records. The record belongs to the
    NEXT condition, and moving it inside the init guard would record only the
    first item per key."""
    _write(tmp_path, "mod.py", '''
def run(rows, seen_senses):
    for row in rows:
        norm_form, sense_id = row
        if norm_form not in seen_senses:
            seen_senses[norm_form] = set()
        if sense_id in seen_senses[norm_form]:
            continue
        seen_senses[norm_form].add(sense_id)
        process(sense_id)
''')
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_ignores_the_seen_set_idiom(tmp_path: Path):
    """``if sid in seen: report_duplicate(); seen.add(sid)`` -- the set tracks
    everything encountered and the `if` REPORTS a repeat rather than gating the
    record. Moving the record inside deletes the duplicate detection."""
    _write(tmp_path, "mod.py", '''
def run(sids):
    seen = set()
    dups = []
    for sid in sids:
        if sid in seen:
            dups.append(sid)
        seen.add(sid)
    return dups
''')
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


def test_effect_flag_outside_its_effect_still_flags_a_report_then_record_defect(tmp_path: Path):
    """Non-vacuousness for the exemption above. A guard whose body only appends
    to an error list is ALSO the shape of the real defect -- record success even
    though the branch failed. What separates the two is whether the guard
    interrogates the same container the record writes to. Here it does not."""
    _write(tmp_path, "mod.py", '''
def run(items, errors, processed):
    for item in items:
        if item.is_broken:
            errors.append(item)
        processed.add(item.id)
''')
    assert len(scan_effect_flag_outside_its_effect(tmp_path)) == 1


def test_effect_flag_outside_its_effect_ignores_list_building(tmp_path: Path):
    """`.append` on a list is ordinary accumulation; it gave 44 of this rule's 50 first hits with
    no success record among them."""
    _write(
        tmp_path,
        "out.py",
        """
def go(rows, out):
    for row in rows:
        if row.ok:
            process(row)
        out.append(row)
""",
    )
    assert scan_effect_flag_outside_its_effect(tmp_path) == []


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


# ---- sql_selects_unread_column -------------------------------------------


def test_sql_selects_unread_column_flags_the_canonical_case(tmp_path: Path):
    """Four columns fetched, four bound, one never read -- the quiet shape that ships."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload, updated_at FROM jobs WHERE ts > 1"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    findings = scan_sql_selects_unread_column(tmp_path)
    assert len(findings) == 1, findings
    assert "updated_at" in findings[0].detail


def test_sql_selects_unread_column_accepts_an_underscored_binding(tmp_path: Path):
    """`_` is how a deliberately-ignored column is spelled, and it is not a defect."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload, updated_at FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, _updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_declines_a_star_select(tmp_path: Path):
    """`SELECT *` names no columns, so there is nothing to compare the unpacking against."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT * FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_declines_two_queries_in_one_function(tmp_path: Path):
    """With two SELECTs it cannot say which unpacking belongs to which, and a coin flip here
    would be worse than silence."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL_A = "SELECT id, uid, payload, updated_at FROM jobs"
SQL_B = "SELECT id, uid FROM clients"

def load(cur):
    cur.execute(SQL_A)
    for job_id, uid, payload, updated_at in cur:
        handle(job_id, uid, payload)
    cur.execute(SQL_B)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


def test_sql_selects_unread_column_accepts_every_column_read(tmp_path: Path):
    """The correct form, which must stay silent."""
    _write(
        tmp_path,
        "out.py",
        '''
SQL = "SELECT id, uid, payload FROM jobs"

def load(cur):
    cur.execute(SQL)
    for job_id, uid, payload in cur:
        handle(job_id, uid, payload)
''',
    )
    assert scan_sql_selects_unread_column(tmp_path) == []


# ---- count_then_fetch_same_table -----------------------------------------


def test_count_then_fetch_same_table_flags_the_canonical_case(tmp_path: Path):
    """Two round trips for one answer: `len(rows)` already is the count."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM jobs WHERE stale")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs WHERE stale")
    return total, cur.fetchall()
''',
    )
    findings = scan_count_then_fetch_same_table(tmp_path)
    assert len(findings) == 1, findings
    assert "jobs" in findings[0].detail


def test_count_then_fetch_same_table_accepts_a_paginated_fetch(tmp_path: Path):
    """A LIMIT is the one legitimate reason to ask twice: the page does not carry the total."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM jobs WHERE stale")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs WHERE stale LIMIT 100")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_accepts_a_grouped_count(tmp_path: Path):
    """A GROUP BY answers a breakdown the fetched rows do not contain."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
    by_status = cur.fetchall()
    cur.execute("SELECT id, uid FROM jobs")
    return by_status, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_accepts_a_different_table(tmp_path: Path):
    """Counting one table and fetching another is two answers, not one asked twice."""
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur):
    cur.execute("SELECT COUNT(*) FROM clients")
    total = cur.fetchone()[0]
    cur.execute("SELECT id, uid FROM jobs")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


def test_count_then_fetch_same_table_declines_an_interpolated_table(tmp_path: Path):
    """An interpolated table name renders as `?`, not as nothing.

    Dropping it spliced the surrounding text together, the table was read as `where`, and that
    supplied both of this rule's first two hits against real code.
    """
    _write(
        tmp_path,
        "out.py",
        '''
def scan(cur, table):
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE stale")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT id FROM {table} WHERE stale")
    return total, cur.fetchall()
''',
    )
    assert scan_count_then_fetch_same_table(tmp_path) == []


# ---- sentinel_cached_as_answer -------------------------------------------


def test_sentinel_cached_as_answer_flags_the_canonical_case(tmp_path: Path):
    """One transient failure pins the key to None for the lifetime of the process."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def lookup(key):
    if key not in _cache:
        try:
            _cache[key] = fetch(key)
        except Exception:
            _cache[key] = None
    return _cache[key]
""",
    )
    findings = scan_sentinel_cached_as_answer(tmp_path)
    assert len(findings) == 1, findings
    assert "None" in findings[0].detail


def test_sentinel_cached_as_answer_flags_an_empty_container(tmp_path: Path):
    """`{}` cached on a build failure is the same defect wearing a different sentinel."""
    _write(
        tmp_path,
        "out.py",
        """
_map_cache = {}

def maps(src):
    try:
        _map_cache[src] = build(src)
    except Exception:
        _map_cache[src] = {}
    return _map_cache[src]
""",
    )
    assert len(scan_sentinel_cached_as_answer(tmp_path)) == 1


def test_sentinel_cached_as_answer_ignores_a_real_value(tmp_path: Path):
    """A handler that caches a genuine fallback is not caching a failure."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def lookup(key):
    try:
        _cache[key] = fetch(key)
    except Exception:
        _cache[key] = DEFAULT_FOR[key]
    return _cache[key]
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []


def test_sentinel_cached_as_answer_ignores_a_plain_local(tmp_path: Path):
    """Assigning None to something that is not a cache costs nothing after the call returns."""
    _write(
        tmp_path,
        "out.py",
        """
def lookup(key, results):
    try:
        results[key] = fetch(key)
    except Exception:
        results[key] = None
    return results[key]
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []


def test_sentinel_cached_as_answer_ignores_a_write_outside_a_handler(tmp_path: Path):
    """Caching None on a path that did not fail says nothing about a swallowed error."""
    _write(
        tmp_path,
        "out.py",
        """
_cache = {}

def reset(key):
    _cache[key] = None
""",
    )
    assert scan_sentinel_cached_as_answer(tmp_path) == []

HELPER = '\nclass Stats:\n    def _inc_stat(self, key, delta=1):\n        with self._lock:\n            self.stats[key] += delta\n\n    def use(self):\n        self._inc_stat("pages")\n'

# ---- accumulator_helper_bypassed -----------------------------------------


def test_accumulator_helper_bypassed_flags_a_sibling_module(tmp_path: Path):
    """The canonical shape: `_inc_stat` in a mixin, a direct `+=` in its sibling.

    Package-wide on purpose -- a per-file rule saw a helper with no bypasses and bypasses with no
    helper, and reported nothing on the very defect it was written for.
    """
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "parallel_mixin.py",
        """
class Parallel:
    def paginate(self, ids):
        self.stats["total_paginated"] += len(ids)
""",
    )
    findings = scan_accumulator_helper_bypassed(tmp_path)
    assert len(findings) == 1, findings
    assert "_inc_stat" in findings[0].detail


def test_accumulator_helper_bypassed_accepts_a_write_under_the_lock(tmp_path: Path):
    """The helper here is a lock plus the write, so a caller already holding it skips nothing."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "parallel_mixin.py",
        """
class Parallel:
    def paginate(self, ids):
        with self._lock:
            self.stats["total_paginated"] += len(ids)
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_accepts_assigning_a_measurement(tmp_path: Path):
    """`stats["root_total_count"] = count` stores a value just computed; routing it through an
    incrementing helper would be wrong, not safer. All four surviving hits in one codebase were
    this shape."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def crawl(self, count):
        self.stats["root_total_count"] = count
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_accepts_a_test_fixture(tmp_path: Path):
    """A test arranging state before asserting on it is building a fixture, not bypassing."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "test_crawler.py",
        """
def test_pagination(c):
    c.stats["total_paginated"] += 1
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_ignores_a_local_accumulator(tmp_path: Path):
    """A local is shared with nobody, so writing it directly bypasses nothing.

    Two earlier versions of this test proved nothing and were replaced: one used
    `findings.append(...)`, which stopped being a candidate at all once the rule narrowed to
    accumulation, and one defined a helper no other function called, which the rule skips before
    it ever reaches the shared-structure question.
    """
    _write(
        tmp_path,
        "scanner.py",
        """
def collect(key):
    counts = {}
    counts[key] += 1
    return counts


def other():
    counts = {}
    counts["fixed"] += 1
    return counts


def run(key):
    return collect(key), other()
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_needs_a_parameter_keyed_owner(tmp_path: Path):
    """With every writer using a literal key there is no helper, and nobody bypasses anything.

    `record` here mutates two structures, so it can never be an owner itself -- which is what
    makes it the site the rule WOULD report if it accepted a literal-keyed writer as the owner.
    """
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def paginate(self, ids):
        self.stats["total_paginated"] += len(ids)

    def record(self, ids, seen):
        self.stats["total_dup"] += len(ids)
        self.audit_log["last"] = seen

    def run(self, ids, seen):
        self.paginate(ids)
        self.record(ids, seen)
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


# ---- test_asserts_against_production_constant ----------------------------


def test_asserts_against_production_constant_flags_a_rederived_expectation(tmp_path: Path):
    """The expected value is computed from the constant the code under test reads.

    Found in the scraper codebase in its purest form: `expected = min_days * MULT` followed by
    `assert expected == min_days * MULT` -- a variable compared with its own definition, which
    never called production code at all.
    """
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY, compute_backoff


def test_backoff():
    assert compute_backoff(3) == BASE_DELAY * 2 ** 3
""",
    )
    findings = scan_test_asserts_against_production_constant(tmp_path)
    assert len(findings) == 1, findings
    assert "BASE_DELAY" in findings[0].detail


def test_asserts_against_production_constant_accepts_a_literal(tmp_path: Path):
    """The form this rule asks for: change the constant and the test has to change too."""
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY, compute_backoff


def test_backoff():
    assert compute_backoff(3) == 24
    assert BASE_DELAY == 3
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_accepts_a_boundary_argument(tmp_path: Path):
    """A constant feeding the code under test is not standing in for its answer.

    `choose_backend(MIN_ROWS - 1, ...) == "numpy"` is a correct boundary test whose expected value
    is the literal on the other side, and that shape was six of this rule's first sixteen hits.
    """
    _write(
        tmp_path,
        "test_dispatch.py",
        """
from kernels import MIN_ROWS, choose_backend


def test_below_the_floor():
    assert choose_backend(MIN_ROWS - 1) == "numpy"
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_accepts_a_path_join(tmp_path: Path):
    """`CHECKPOINT_DIR / "x.jsonl"` is path construction wearing an operator, not arithmetic."""
    _write(
        tmp_path,
        "test_checkpoint.py",
        """
from rescan import CHECKPOINT_DIR, checkpoint_path


def test_path():
    assert checkpoint_path("01abc") == CHECKPOINT_DIR / "~01abc.jsonl"
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_ignores_a_local_constant(tmp_path: Path):
    """A constant the test defines for itself is the expected value, spelled once."""
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import compute_backoff

BASE = 3


def test_backoff():
    assert compute_backoff(3) == BASE * 2 ** 3
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_ignores_two_derivations(tmp_path: Path):
    """Both sides derived from the same constant is a different (and rarer) mistake.

    Reporting it here would blur the message, which is about an expected value that moves with the
    implementation -- here neither side is the thing under test.
    """
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY


def test_algebra():
    assert BASE_DELAY * 4 == BASE_DELAY * 2 * 2
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_only_reads_test_files(tmp_path: Path):
    """Production code deriving one constant from another is ordinary, not a broken assertion."""
    _write(
        tmp_path,
        "throttling.py",
        """
from config import BASE_DELAY

MAX_DELAY = BASE_DELAY * 60


def check():
    assert MAX_DELAY == BASE_DELAY * 60
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []
