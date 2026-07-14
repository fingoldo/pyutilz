"""Unit tests for pyutilz.dev.code_audit AST scanners.

Each scanner gets a positive case (constructed snippet that MUST be
flagged) and a negative case (constructed snippet that MUST NOT be
flagged). Tests use tmp_path so the audit runs against a hermetic
directory; no cross-test bleed.
"""
from __future__ import annotations

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
    from pyutilz.dev.code_audit.silent_escalation import scan_log_only_except as _sloe, DEFAULT_ESCALATION_ATTRS as _dea
    from pyutilz.dev.code_audit.sql_migrations import scan_sql_migration_idempotency as _ssmi
    from pyutilz.dev.code_audit.registry import run_all as _ra, SCANNERS as _sc
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
    assert facade.DEFAULT_ESCALATION_ATTRS is _dea
    assert facade.scan_sql_migration_idempotency is _ssmi
    assert facade.run_all is _ra
    assert facade.SCANNERS is _sc
    assert facade.main is _main
    # Every scanner in the registry is the facade-level attribute of the same name.
    for name, fn in facade.SCANNERS.items():
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
