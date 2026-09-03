"""Scanner tests for spy_arity, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_stale_test_spy_arity,
)

from ._helpers import _write

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
    """A production call site in attribute form (`prod_module.build_rows(...)`) must be matched
    the same as a bare-Name call site.

    The receiver has to RESOLVE TO THE PATCHED MODULE. A same-named method on an unrelated object
    is not the patched function, and counting it made the scanner contradict its own documented
    "false negatives are the safe failure mode here, not false positives" -- see the sibling
    `..._ignores_a_same_named_method_on_another_class` test.
    """
    _write(tmp_path, "prod_module.py", """
def build_rows(tables, cid, node, memo=None):
    pass
""")
    _write(tmp_path, "caller.py", """
import prod_module

class Caller:
    def run(self):
        prod_module.build_rows(1, 2, 3, 4)
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


# ---- F99: spy arity, cross-class short-name collision -------------------------------


def test_stale_test_spy_arity_ignores_a_same_named_method_on_another_class(tmp_path: Path):
    _write(tmp_path, "prod.py", "def build_rows(a):\n    return a\n")
    _write(tmp_path, "caller.py", """
import prod


def go(a):
    return prod.build_rows(a)
""")
    _write(tmp_path, "other.py", """
class Other:
    def build_rows(self, a, b, c):
        return a


def use(o):
    return o.build_rows(1, 2, 3)
""")
    _write(tmp_path, "test_x.py", """
from unittest.mock import patch


def spy(a):
    return a


def test_it():
    with patch("prod.build_rows", side_effect=spy):
        pass
""")
    assert scan_stale_test_spy_arity(tmp_path) == []
