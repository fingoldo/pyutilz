"""Tests for pyutilz.dev.block_extract: planning and applying statement-range extraction with exact data flow."""

import importlib.util
import sys
import textwrap

import pytest

from pyutilz.dev.block_extract import apply_extraction, name_events, plan_extraction


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return p


def _line(p, needle):
    return next(i for i, ln in enumerate(p.read_text().splitlines(), 1) if needle in ln)


def test_accumulator_on_one_line_is_an_input():
    ev = name_events([__import__("ast").parse("x = [i for i in x if i]").body[0]])
    assert ev[0][1:] == ("load", "x") and ev[-1][1:] == ("store", "x")


def test_inputs_outputs_and_passthrough(tmp_path):
    p = _write(
        tmp_path,
        "m.py",
        """\
        def f(a, b):
            x = a + 1
            y = x * b
            z = y + a
            return z + y
        """,
    )
    plan = plan_extraction(p, "f", 3, 4)
    assert plan.inputs == ["x", "b", "a"] and plan.outputs == ["y", "z"] and not plan.problems


def test_comprehension_and_handler_names_are_not_data_flow(tmp_path):
    p = _write(
        tmp_path,
        "m.py",
        """\
        def f(xs):
            e = 0
            try:
                s = sum(v for v in xs)
            except ValueError as e:
                print(e)
                s = 0
            return s, e
        """,
    )
    plan = plan_extraction(p, "f", 3, 7)
    assert plan.inputs == ["xs", "e"] or plan.inputs == ["xs"]
    assert "v" not in plan.inputs and plan.outputs == ["s"]


def test_refusals(tmp_path):
    p = _write(
        tmp_path,
        "m.py",
        """\
        def f(xs):
            for x in xs:
                if x:
                    break
                y = x
            if xs:
                z = 1
            return z
        """,
    )
    assert any("Break" in m for m in plan_extraction(p, "f", 3, 4).problems)
    assert any("only on some paths" in m for m in plan_extraction(p, "f", 6, 7).problems)
    assert any("Return" in m for m in plan_extraction(p, "f", 8, 8).problems)


def test_inside_a_loop_the_next_iteration_counts(tmp_path):
    p = _write(
        tmp_path,
        "m.py",
        """\
        def f(n):
            acc = 0
            last = None
            for i in range(n):
                if last is not None:
                    acc += last
                last = i * 2
            return acc
        """,
    )
    plan = plan_extraction(p, "f", 7, 7)
    assert plan.in_loop and plan.outputs == ["last"]


def _import(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)  # restored after the test
    spec.loader.exec_module(mod)
    return mod


def test_apply_keeps_behaviour_and_comments(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "pk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    core = _write(
        pkg,
        "core.py",
        """\
        import math

        LIMIT = 3


        def f(values):
            total = 0
            # keep this comment
            for v in values:
                if v > LIMIT:
                    total += math.floor(v)
            scaled = total * 2
            return scaled
        """,
    )
    expected = _import(core, "pk.core", monkeypatch).f([1, 5.5, 7.2])
    plan = plan_extraction(core, "f", 8, 11)
    result = apply_extraction(plan, "_sum_big", pkg / "stages.py", "pk.stages", core_module="pk.core")
    assert "# keep this comment" in result.helper_source  # comments move with their code
    assert result.call.strip() == "total = _sum_big(values, total)"
    for m in ("pk.core", "pk.stages"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    stages = _import(pkg / "stages.py", "pk.stages", monkeypatch)
    assert stages.math is __import__("math") and stages.LIMIT == 3  # the helper's module imports what it reads
    assert _import(core, "pk.core", monkeypatch).f([1, 5.5, 7.2]) == expected


def test_state_mode_rewrites_names_but_not_attributes_or_keywords(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "pq"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    core = _write(
        pkg,
        "core.py",
        """\
        from types import SimpleNamespace


        def g(n):
            st = SimpleNamespace(recipes={}, count=0)
            for i in range(n):
                st.recipes[i] = dict(count=i)  # keyword 'count' must stay
                st.count += 1
            return st.count, len(st.recipes)
        """,
    )
    # the state names here are attribute-accessed already; plan the loop and apply with no state names
    plan = plan_extraction(core, "g", 6, 8)
    assert plan.inputs == ["n", "st"] and not plan.problems
    from pyutilz.dev.block_extract import _to_state

    text = "recipes[k] = dict(count=count)  # count stays a keyword\nobj.count = count + 1\n"
    assert _to_state(text, {"recipes", "count"}, "st") == "st.recipes[k] = dict(count=st.count)  # count stays a keyword\nobj.count = st.count + 1\n"


def test_relative_imports_in_moved_code_become_absolute():
    from pyutilz.dev.block_extract import absolutise_relative_imports

    text = "    from .._helpers import a\n    from . import b\n"
    assert absolutise_relative_imports(text, "pkg.sub.core") == "    from pkg._helpers import a\n    from pkg.sub import b\n"


def test_apply_refuses_a_plan_with_problems(tmp_path):
    p = _write(tmp_path, "m.py", "def f(x):\n    if x:\n        return 1\n    return 2\n")
    with pytest.raises(ValueError):
        apply_extraction(plan_extraction(p, "f", 2, 3), "_h", tmp_path / "s.py", "s", core_module="m")


def test_rename_in_function_rewrites_locals_to_attributes(tmp_path, monkeypatch):
    from pyutilz.dev.block_extract import rename_in_function

    p = _write(tmp_path, "m.py", """\
        from types import SimpleNamespace


        def f(n):
            recipes = SimpleNamespace()
            _a_pre_recipes = {}  # registry of family a
            for i in range(n):
                _a_pre_recipes[i] = dict(_a_pre_recipes=i)
            total = len(_a_pre_recipes)
            return total, other(_a_pre_recipes)


        def other(_a_pre_recipes):
            return _a_pre_recipes
        """)
    n = rename_in_function(p, "f", {"_a_pre_recipes": "recipes.a"})
    src = p.read_text()
    assert n == 4 and "recipes.a = {}  # registry of family a" in src and "dict(_a_pre_recipes=i)" in src
    assert "def other(_a_pre_recipes):" in src  # other functions untouched
    assert _import(p, "renamed_m", monkeypatch).f(3) == (3, {0: {"_a_pre_recipes": 0}, 1: {"_a_pre_recipes": 1}, 2: {"_a_pre_recipes": 2}})


def test_rename_refuses_closures_fstrings_and_params(tmp_path):
    from pyutilz.dev.block_extract import rename_in_function

    p = _write(tmp_path, "m.py", "def f(a):\n    b = 1\n    g = lambda: b\n    s = f'{b}'\n    return g, s\n")
    for mapping in ({"a": "st.a"}, {"b": "st.b"}):
        with pytest.raises(ValueError):
            rename_in_function(p, "f", mapping)
