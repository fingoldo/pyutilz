"""Regression tests for the 2026-09-03 typing/lint-hygiene audit (audits/2026-09-03/06-typing-lint-hygiene.md).

One test (or one small cluster) per finding that changed behaviour. Each asserts the BEHAVIOUR the
finding described as broken, so a revert fails here rather than silently reappearing.
"""

import ast
import subprocess  # nosec B404 - runs the repo's own pin-check script with a fixed argv, no shell
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "pyutilz"


# ---- F01: the one identical stamped justification is gone -------------------------------------


def test_f01_no_batch_stamped_type_ignore_justification():
    """No `# type: ignore` in src carries the batch-applied "json/external lib/dynamic attr" reason.

    That single string was pasted verbatim onto 54 sites and was demonstrably false at several of
    them (numpy `.nbytes` arithmetic, a locally-built set). Every remaining ignore names its own cause.
    """
    stamped = [
        f"{path.relative_to(SRC_ROOT).as_posix()}:{i}"
        for path in SRC_ROOT.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "untyped upstream source (json/external lib/dynamic attr)" in line
    ]
    assert stamped == [], f"batch-stamped type-ignore justification is back at: {stamped}"


def test_f01_sparse_memory_usage_returns_a_real_int():
    """`get_sparse_memory_usage` returns a builtin int, not numpy's np.intp, which is what the removed ignore was hiding."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from pyutilz.core.matrix import get_sparse_memory_usage

    mat = scipy_sparse.csr_matrix([[1, 0], [0, 2]])
    result = get_sparse_memory_usage(mat)
    assert type(result) is int


# ---- F02: the duplicate scanner pair is deliberate and says so ---------------------------------


def test_f02_both_reexport_patch_scanners_acknowledge_each_other():
    """Neither module used to mention the other; a triager could not tell one finding from two."""
    from pyutilz.dev.code_audit import patch_target_is_a_reexport, reexport_patch_target

    assert "reexport_patch_target.py" in (patch_target_is_a_reexport.__doc__ or "")
    assert "patch_target_is_a_reexport.py" in (reexport_patch_target.__doc__ or "")


def test_f02_only_one_of_the_pair_runs_by_default():
    """Both stay registered (so the emitted check id is re-runnable), but only one is on by default."""
    from pyutilz.dev.code_audit.registry import OPT_IN_ONLY, get_scanners

    scanners = get_scanners()
    assert "patch_target_is_a_reexport" in scanners
    assert "reexport_patch_target" in scanners
    assert "reexport_patch_target" in OPT_IN_ONLY
    assert "patch_target_is_a_reexport" not in OPT_IN_ONLY


def test_f02_dotted_helper_is_the_shared_one():
    """The scanner's private `_dotted` copy of `_base.dotted_name` is gone (duplicate_function_body)."""
    from pyutilz.dev.code_audit import patch_target_is_a_reexport

    assert not hasattr(patch_target_is_a_reexport, "_dotted")
    assert patch_target_is_a_reexport.dotted_name.__module__.endswith("_base")


# ---- F03 / F04 / F08: functions that fell off the end ------------------------------------------


def test_f03_check_if_pg_table_exists_reports_unknown_as_none(monkeypatch):
    """A falsy safe_execute result is "I could not find out", explicitly None rather than a fall-off."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.schema import check_if_pg_table_exists

    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: [])
    assert check_if_pg_table_exists("t") is None
    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: [[True]])
    assert check_if_pg_table_exists("t") is True


def test_f03_ensure_pg_table_exists_refuses_to_create_on_an_unknown_probe(monkeypatch):
    """ "Unknown" must not be read as "absent" and followed by a CREATE against a database that could not be queried."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.schema import ensure_pg_table_exists

    monkeypatch.setattr(facade, "check_if_pg_table_exists", lambda *a, **k: None)
    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: pytest.fail("CREATE must not run on an unknown probe"))
    with pytest.raises(RuntimeError, match="Could not determine whether table"):
        ensure_pg_table_exists("mytable")


def test_f04_get_table_fields_never_returns_none(monkeypatch):
    """Its result is concatenated into SQL, so None (or the literal text "None") must be impossible."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.execution import get_table_fields

    class _Cur:
        description = None

        def execute(self, *a, **k):
            return None

        def fetchall(self):
            return []

    monkeypatch.setattr(facade, "get_cursor_type", lambda *a, **k: "c")
    monkeypatch.setattr(facade, "get_cursor", lambda *a, **k: _Cur())
    with pytest.raises(RuntimeError, match="no cursor description"):
        get_table_fields("orders", "o")


def test_f08_objects_dumper_process_object_returns_false_not_none(tmp_path):
    """`process_objects` counts these results; None worked only by being falsy."""
    from pyutilz.core.pythonlib.filesystem import ObjectsDumper, ObjectsLoader

    dumper = ObjectsDumper()
    assert dumper._process_object({"a": None}, "a", str(tmp_path / "a.dump")) is False
    loader = ObjectsLoader()
    assert loader._process_object({}, "a", str(tmp_path / "missing.dump")) is False


def test_f08_basic_db_execute_returns_an_empty_list_for_a_no_result_statement(monkeypatch):
    """The docstring promises an empty list for a statement with no result set; a DuplicateTable
    collision (a CREATE that already exists) is exactly that case and used to return a bare None,
    so `for row in basic_db_execute(...)` raised TypeError on a path the documentation handles."""
    from psycopg2.errors import DuplicateTable

    import pyutilz.database.db as facade
    from pyutilz.database.db.execution import basic_db_execute

    class _Cur:
        def execute(self, *a, **k):
            raise DuplicateTable("relation already exists")

    monkeypatch.setattr(facade, "get_cursor_type", lambda *a, **k: "c")
    monkeypatch.setattr(facade, "get_cursor", lambda *a, **k: _Cur())
    assert basic_db_execute("execute", "create table t (id int)") == []


def test_f08_basic_db_execute_has_no_implicit_none_terminal():
    """Its retry loop can fall through (named-cursor collisions exhausting the budget); the terminal
    is an explicit raise naming the budget, not an implicit None a caller reads as "no rows"."""
    import inspect

    from pyutilz.database.db import execution

    tree = ast.parse(inspect.getsource(execution))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "basic_db_execute")
    assert isinstance(func.body[-1], ast.Raise)


# ---- F05: concrete parameter type instead of `object` ------------------------------------------


def test_f05_add_weighted_aggregates_declares_a_selector():
    """`object` supports no `-`; the signature type-checked only because the other operand was Any."""
    cs = pytest.importorskip("polars.selectors")
    from pyutilz.data.polarslib.aggregations import add_weighted_aggregates

    assert add_weighted_aggregates.__annotations__["columns_selector"] is cs.Selector


# ---- F06: the blocking ruff gate is pinned in practice, not just on paper -----------------------


def test_f06_pin_check_script_reports_a_mismatch():
    """`language: system` means the hook runs whatever ruff is installed; this gate is what says so."""
    script = REPO_ROOT / "scripts" / "check_pinned_tool_versions.py"
    assert script.is_file()
    out = subprocess.run(  # nosec B603 - fixed argv (sys.executable plus a repo path), shell=False
        [sys.executable, str(script)], capture_output=True, text=True, check=False, cwd=str(REPO_ROOT)
    )
    # Either the box matches the pin (exit 0, no output) or it does not and the mismatch is NAMED.
    if out.returncode != 0:
        assert "pinned-tool-version mismatch" in out.stdout


def test_f06_pin_check_is_wired_as_a_hook():
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "check_pinned_tool_versions.py" in text


# ---- F07: the strict-mode beachhead grew --------------------------------------------------------


def test_f07_beachhead_includes_the_two_new_subpackages():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pyutilz.web.proxy.*"' in text
    assert '"pyutilz.llm.openrouter_provider.*"' in text


# ---- F09: _benchmarks is linted ------------------------------------------------------------------


def test_f09_benchmarks_is_no_longer_excluded_from_ruff():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert 'exclude = [".git", "__pycache__", "build", "dist"]' in pyproject
    assert '"_benchmarks/**"' in pyproject
    assert "exclude: (^|/)_benchmarks/" not in precommit


# ---- F10: the unsafe attribute access is gone ----------------------------------------------------


def test_f10_end_line_survives_a_node_without_lineno():
    """`ast.AST` genuinely has no `lineno`; the old fallback reached for it unguarded, mid-rewrite."""
    from pyutilz.dev.freevar_analysis import _end_line

    assert _end_line(ast.Load()) == 0
    assert _end_line(ast.parse("x = 1").body[0]) == 1


def test_f10_platform_ignores_name_their_environment():
    """The Windows-only / numba ignores stay (closed decision) but must say WHICH environment."""
    bare = []
    seen = 0
    for rel in ("system/system/memory.py", "system/system/misc.py", "text/similarity/_numba_kernels.py"):
        for line in (SRC_ROOT / rel).read_text(encoding="utf-8").splitlines():
            if "# type: ignore[attr-defined]" in line:
                seen += 1
                if not line.split("# type: ignore[attr-defined]", 1)[1].strip():
                    bare.append(f"{rel}: {line.strip()}")
    # Assert the precondition too: if the ignores were all deleted (the stronger failure, since the
    # closed decision says they must stay), an all-conditional check would pass by seeing nothing.
    assert seen == 8, f"expected the 8 platform-conditional attr-defined ignores, found {seen}"
    assert bare == [], f"attr-defined ignore(s) with no environment note: {bare}"


# ---- F11: no invisible characters anywhere in the tree --------------------------------------------


# Built from code points, never written literally: a literal invisible character here would make this test fail on its own source.
_INVISIBLE = tuple(chr(cp) for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))


def test_f11_no_invisible_characters_in_python_sources():
    """A U+200B inside a docstring is undiagnosable by reading the file, and docstrings are load-bearing here."""
    offenders = []
    for base in ("src", "tests", "scripts", "_benchmarks"):
        for path in (REPO_ROOT / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for ch in _INVISIBLE:
                if ch in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {ch!r}")
    assert offenders == [], f"invisible character(s) in source: {offenders}"


def test_f11_invisible_character_rules_are_selected():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for code in ("PLE2510", "PLE2512", "PLE2513", "PLE2514", "PLE2515", "RET503"):
        assert f'"{code}"' in text


# ---- F12: get_attr is usable and its guard is reachable --------------------------------------------


def test_f12_get_attr_accepts_none_and_returns_the_default():
    """The `unwanted_value` guard exists to catch exactly this; `obj: dict` said it could not fire."""
    from pyutilz.core.pythonlib.objects import get_attr

    assert get_attr(None, "hosts") == []
    assert get_attr(None, "hosts", default_value="x") == "x"
    # A non-None unwanted_value used to walk straight into None.get(...).
    assert get_attr(None, "hosts", unwanted_value="sentinel") == []
    assert get_attr({"hosts": [1, 2]}, "hosts") == [1, 2]


def test_f12_get_attr_return_is_not_object():
    from pyutilz.core.pythonlib.objects import get_attr

    assert get_attr.__annotations__["return"] is not object


# ---- deferred: the C901 ratchet split ---------------------------------------------------------------


def test_cli_stream_consumer_is_a_separate_function():
    """`_generate_cli` was split at 25 complexity; the ratchet only turns down."""
    import queue

    from pyutilz.llm.claude_code_provider import _consume_cli_stream

    q: "queue.Queue" = queue.Queue()
    q.put('{"type": "system", "subtype": "init"}')
    q.put('{"type": "result", "subtype": "success", "result": "hello"}')
    assert _consume_cli_stream(q, timeout=5.0) == ("hello", None, False)

    q2: "queue.Queue" = queue.Queue()
    q2.put('{"type": "result", "subtype": "error", "error": "nope"}')
    assert _consume_cli_stream(q2, timeout=5.0) == (None, "nope", False)

    q3: "queue.Queue" = queue.Queue()
    q3.put(None)  # reader EOF sentinel: neither a result nor a timeout
    assert _consume_cli_stream(q3, timeout=5.0) == (None, None, False)
