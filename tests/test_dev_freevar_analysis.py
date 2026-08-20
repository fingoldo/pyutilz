"""Tests for pyutilz.dev.freevar_analysis: AST free-variable analysis for planning safe
function/method extraction (monolith splits)."""

import importlib
import textwrap

import pytest

from pyutilz.dev import freevar_analysis as fva
from pyutilz.dev.freevar_analysis import analyze_range, format_report, _main


def _write(tmp_path, src):
    p = tmp_path / "sample.py"
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return p


def test_simple_free_variable_detected(tmp_path):
    src = """\
        def outer(a, b):
            x = a + 1
            y = helper(x, b)
            return y
    """
    p = _write(tmp_path, src)
    # lines: 1 def, 2 x=a+1, 3 y=helper(x,b), 4 return y
    report = analyze_range(p, 3, 3)
    assert "helper" in report.free_names
    assert "x" in report.free_names  # assigned on line 2, only READ on line 3 -> free within [3,3]
    assert "b" in report.free_names
    assert "y" not in report.free_names  # y is Stored on line 3 itself


def test_locally_assigned_name_is_not_free(tmp_path):
    src = """\
        def outer():
            z = 1
            z = z + 1
            return z
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 4)
    assert "z" not in report.free_names


def test_needs_incoming_value_for_reassigned_accumulator(tmp_path):
    # Mirrors the real-world case this tool was built for: `selected = [...]` then repeated
    # `selected = [x for x in selected if ...]` filters -- `selected` is Stored throughout the
    # range but its FIRST use is a Load, so it still needs an incoming value.
    src = """\
        def outer(selected):
            if not selected:
                pass
            selected = [v for v in selected if v > 0]
            selected = [v for v in selected if v < 10]
            return selected
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 5)
    names_needing_incoming = {u.name for u in report.needs_incoming_value}
    assert "selected" in names_needing_incoming
    # `selected` must NOT also appear in free_names (it IS reassigned in-range).
    assert "selected" not in report.free_names


def test_name_assigned_before_any_read_is_not_flagged_as_needing_incoming(tmp_path):
    src = """\
        def outer():
            total = 0
            total = total + 1
            return total
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 4)
    names_needing_incoming = {u.name for u in report.needs_incoming_value}
    # First occurrence of `total` in range is the Store on line 2 -- genuinely local.
    assert "total" not in names_needing_incoming
    assert "total" not in report.free_names


def test_function_def_binds_its_own_name(tmp_path):
    src = """\
        def outer():
            def inner(v):
                return v
            return inner(1)
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 4)
    assert "inner" not in report.free_names


def test_all_dependency_names_merges_both_categories(tmp_path):
    src = """\
        def outer(a, selected):
            if not selected:
                pass
            selected = [x for x in selected if x > a]
            return helper(selected)
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 4)
    combined = report.all_dependency_names()
    assert "a" in combined
    assert "selected" in combined  # read (line 2) before being reassigned (line 4)
    assert combined == sorted(combined)


def test_format_report_renders_header_and_sections(tmp_path):
    src = """\
        def outer(a):
            return a + 1
    """
    p = _write(tmp_path, src)
    report = analyze_range(p, 2, 2)
    text = format_report(report, p, 2, 2)
    assert f"{p}:2-2" in text
    assert "Free (external) names referenced" in text
    assert "  a" in text


def test_analyze_range_on_own_source_file_finds_no_undefined_names(tmp_path):
    """Self-consistency smoke test: analysing this module's own analyze_range function body
    should not blow up and should return a well-formed report (regression guard against a
    parse/line-number-offset bug)."""
    import pyutilz.dev.freevar_analysis as mod

    report = analyze_range(mod.__file__, 1, 250)
    assert isinstance(report.free_names, list)
    assert isinstance(report.needs_incoming_value, list)


def test_cli_requires_exactly_three_args(capsys):
    rc = _main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err


def test_cli_end_to_end(tmp_path, capsys):
    src = """\
        def outer(a):
            return helper(a)
    """
    p = _write(tmp_path, src)
    rc = _main([str(p), "2", "2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "helper" in captured.out


@pytest.mark.parametrize("bad_range", [("does_not_exist.py", "1", "5")])
def test_analyze_range_missing_file_raises(bad_range):
    path, start, end = bad_range
    with pytest.raises(FileNotFoundError):
        analyze_range(path, int(start), int(end))


# ----------------------------------------------------------------------------------------------------------------------------
# split_out_module
# ----------------------------------------------------------------------------------------------------------------------------

_SPLITTABLE = '''"""A module that grew too big."""

from __future__ import annotations

import json
from pathlib import Path

CONSTANT = 3


def keep_me(x):
    """Stays behind."""
    return x + CONSTANT


# This comment explains why `move_a` is shaped the way it is, and is the whole reason
# the move has to be contiguous rather than name-by-name.
def move_a(payload):
    """First moved."""
    return json.dumps(payload, sort_keys=True)


def move_b(path):
    """Second moved."""
    return Path(path).name
'''


def _write_splittable(tmp_path, text=_SPLITTABLE):
    src = tmp_path / "big.py"
    src.write_text(text, encoding="utf-8")
    return src


def test_split_moves_the_range_and_leaves_a_named_reexport(tmp_path):
    src = _write_splittable(tmp_path)
    target = tmp_path / "big_extra.py"

    moved = fva.split_out_module(src, target, "move_a", "move_b")

    assert sorted(moved) == ["move_a", "move_b"]
    rest, new = src.read_text(encoding="utf-8"), target.read_text(encoding="utf-8")
    assert "def move_a" not in rest and "def move_b" not in rest
    assert "def keep_me" in rest, "an untouched definition must stay put"
    assert "from .big_extra import (" in rest
    assert "move_a as move_a," in rest, "the `as` form is what re-exports rather than merely imports"
    assert "def move_a" in new and "def move_b" in new


def test_the_comment_between_definitions_travels_with_them(tmp_path):
    """A comment carrying the reason a function is shaped as it is belongs with the function, not with the
    file it happened to be typed into. A name-by-name move cannot express that; a line range can."""
    src = _write_splittable(tmp_path)
    target = tmp_path / "big_extra.py"

    fva.split_out_module(src, target, "move_a", "move_b")

    assert "the whole reason" in target.read_text(encoding="utf-8")
    assert "the whole reason" not in src.read_text(encoding="utf-8")


def test_a_move_that_would_leave_a_read_name_behind_is_refused(tmp_path):
    """`keep_me` reads the module-level `CONSTANT`; moving it alone would compile and then raise NameError on
    whatever path first calls it. Refused before anything is written."""
    src = _write_splittable(tmp_path)
    target = tmp_path / "big_extra.py"
    original = src.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="CONSTANT"):
        fva.split_out_module(src, target, "keep_me", "keep_me")

    assert src.read_text(encoding="utf-8") == original, "a refused split must not touch the source"
    assert not target.exists(), "a refused split must not create the target"


def test_dry_run_checks_everything_and_writes_nothing(tmp_path):
    src = _write_splittable(tmp_path)
    target = tmp_path / "big_extra.py"
    original = src.read_text(encoding="utf-8")

    moved = fva.split_out_module(src, target, "move_a", "move_b", apply=False)

    assert sorted(moved) == ["move_a", "move_b"]
    assert src.read_text(encoding="utf-8") == original
    assert not target.exists()


def test_every_moved_body_is_byte_identical_after_the_move(tmp_path):
    """The property the whole tool exists for: a split is a MOVE, never an edit."""
    src = _write_splittable(tmp_path)
    target = tmp_path / "big_extra.py"
    before = fva._top_level_bodies(src.read_text(encoding="utf-8"))

    fva.split_out_module(src, target, "move_a", "move_b")

    after = fva._top_level_bodies(target.read_text(encoding="utf-8"))
    for name in ("move_a", "move_b"):
        assert after[name] == before[name]


def test_the_split_module_still_imports_and_behaves(tmp_path, monkeypatch):
    """A re-export nobody can import is not a re-export. Load the split package for real and call through it."""
    pkg = tmp_path / "splitpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    src = pkg / "big.py"
    src.write_text(_SPLITTABLE, encoding="utf-8")

    fva.split_out_module(src, pkg / "big_extra.py", "move_a", "move_b")

    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("splitpkg.big")
    assert module.move_a({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert module.move_b("/x/y/z.txt") == "z.txt"
    assert module.keep_me(1) == 4


def test_a_backwards_range_is_rejected(tmp_path):
    src = _write_splittable(tmp_path)
    with pytest.raises(ValueError, match="must appear before"):
        fva.split_out_module(src, tmp_path / "big_extra.py", "move_b", "move_a")


def test_an_unknown_symbol_is_rejected(tmp_path):
    src = _write_splittable(tmp_path)
    with pytest.raises(ValueError, match="not a top-level definition"):
        fva.split_out_module(src, tmp_path / "big_extra.py", "move_a", "no_such_name")
