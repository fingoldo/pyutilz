"""Scanner tests for near_duplicate_function_body, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_duplicate_conditions,
    scan_near_duplicate_function_body,
)

from ._helpers import _write

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
