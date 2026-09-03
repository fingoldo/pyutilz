"""Scanner tests for source_text_assertions, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.source_text_assertions import scan_source_text_assertions

from ._helpers import _write

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


# ---- F14/F85/F86/F175/F176: source-text assertions ----------------------------------


def test_source_text_assertions_uses_the_relative_path_for_the_tests_check(tmp_path: Path):
    root = tmp_path / "tests" / "proj"
    root.mkdir(parents=True)
    _write(root, "prod.py", """
import inspect


def check():
    src = inspect.getsource(g)
    assert "x" in src
""")
    assert scan_source_text_assertions(root) == []


def test_source_text_assertions_scopes_siblings_independently(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import inspect


def test_outer():
    def check_source():
        src = inspect.getsource(g)
        assert "x" in src

    def check_behaviour():
        src = build()
        assert "x" in src

    check_source()
    check_behaviour()
""")
    findings = scan_source_text_assertions(tmp_path)
    assert len(findings) == 1 and findings[0].line == 7


def test_source_text_assertions_sees_a_pytest_fail_guard(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import inspect
import pytest


def test_x():
    src = inspect.getsource(g)
    if "x" not in src:
        pytest.fail("missing")
""")
    assert len(scan_source_text_assertions(tmp_path)) == 1


def test_source_text_assertions_sees_an_annotated_binding(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import inspect


def test_x():
    src: str = inspect.getsource(g)
    assert "x" in src
""")
    assert len(scan_source_text_assertions(tmp_path)) == 1


def test_source_text_assertions_resolves_a_dis_alias(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import dis as d


def test_x():
    out = d.dis(f)
    assert "LOAD" in out
""")
    assert len(scan_source_text_assertions(tmp_path)) == 1
