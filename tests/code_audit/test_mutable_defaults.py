"""Scanner tests for mutable_defaults, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_mutable_defaults,
    scan_parameter_aliasing_mutation,
)

from ._helpers import _write

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

# ---- F59: mutable default built with arguments --------------------------------------


def test_mutable_defaults_flags_a_call_form_with_arguments(tmp_path: Path):
    _write(tmp_path, "a.py", """
def m(x=dict(a=1)):
    x.update({"b": 2})
    return x
""")
    assert len(scan_mutable_defaults(tmp_path)) == 1


def test_mutable_defaults_ignores_the_none_sentinel_idiom(tmp_path: Path):
    _write(tmp_path, "a.py", """
def m(x=None):
    x = x if x is not None else {}
    return x
""")
    assert scan_mutable_defaults(tmp_path) == []


# ---- parameter_aliasing_mutation: 2026-09-03 downstream-scan precision round ----
#
# A scan of two fresh repos produced 11 findings from this check - the suite's only P0s - and all
# 11 were false. Each shape below is one of them, kept as a negative case.


def test_parameter_aliasing_mutation_scalar_loop_counter_is_clean(tmp_path: Path):
    """`i = l; while i < r: buf[i]...; i += 1` -- the hand-rolled counter every @njit kernel is
    written with. `+=` on a number REBINDS the local; the index and loop-bound uses prove it is a
    number, since a numpy array cannot drive a `while` test."""
    _write(tmp_path, "ok.py", """
def bin_by_value(array, l, r, mask):
    out = []
    i = l
    while i < r:
        out.append(array[mask[i]])
        i += 1
    return out
""")
    assert scan_parameter_aliasing_mutation(tmp_path) == []


def test_parameter_aliasing_mutation_mutually_exclusive_branches_are_clean(tmp_path: Path):
    """The alias is established in the `if` arm and the mutation performed in the `else` arm, so
    the two lines can never both run against the same object."""
    _write(tmp_path, "ok.py", """
def start(tags, parent_run_id=None):
    run_tags = {"parent": parent_run_id} if parent_run_id else None
    if tags:
        if run_tags is None:
            run_tags = tags
        else:
            run_tags.update(tags)
    return run_tags
""")
    assert scan_parameter_aliasing_mutation(tmp_path) == []


def test_parameter_aliasing_mutation_named_output_buffer_is_clean(tmp_path: Path):
    """A parameter NAMED as a caller-owned buffer is one the callee is meant to write into -- that
    is why the caller preallocated it. Writing through is the contract, not a leak."""
    _write(tmp_path, "ok.py", """
def score(values_shared, out, n):
    vals = values_shared
    dest = out
    for i in range(n):
        vals[i] = i
        dest[i] = i
    return vals
""")
    assert scan_parameter_aliasing_mutation(tmp_path) == []


def test_parameter_aliasing_mutation_same_branch_mutation_still_flagged(tmp_path: Path):
    """The branch-exclusivity fix must not eat an alias and a mutation living in the SAME arm --
    the original confirmed bug (`returning_fields = history_fields; returning_fields += [hash]`)
    sits inside an `if`."""
    _write(tmp_path, "bad.py", """
def upsert(history_fields, hash_field, with_hash):
    if with_hash:
        returning_fields = history_fields
        returning_fields += [hash_field]
        return returning_fields
    return history_fields
""")
    findings = scan_parameter_aliasing_mutation(tmp_path)
    assert len(findings) == 1 and findings[0].severity == "P0"
