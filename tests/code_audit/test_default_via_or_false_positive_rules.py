"""Each rule added after the 2026-10-21 measurement on py-ci-shared: the shape it exempts is not flagged, and the real trap next to it still is."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyutilz.dev.code_audit.default_via_or import scan_default_via_or_trap


def _flagged(tmp_path: Path, source: str) -> list[int]:
    (tmp_path / "mod.py").write_text(source, encoding="utf-8")
    return [f.line for f in scan_default_via_or_trap(tmp_path)]


@pytest.mark.parametrize(
    "source",
    [
        'def f(names):\n    return ", ".join(names) or "none"\n',
        'def f(name):\n    return name or "<module>"\n',
        'def f(exc):\n    return exc.msg or f"{exc!r} has no message"\n',
    ],
    ids=["join-with-text-fallback", "name-with-label-fallback", "attribute-with-fstring-fallback"],
)
def test_a_text_fallback_is_not_a_trap(tmp_path, source):
    assert _flagged(tmp_path, source) == []


def test_a_numeric_default_is_still_a_trap(tmp_path):
    assert _flagged(tmp_path, "def f(cfg):\n    return cfg.get('n_jobs') or 4\n") == [2]


def test_two_booleans_joined_by_or_are_not_a_trap(tmp_path):
    source = "def f(a, b):\n    return isinstance(a, int) or isinstance(b, int)\n"
    assert _flagged(tmp_path, source) == []


def test_a_path_on_the_left_is_not_a_trap(tmp_path):
    source = "from pathlib import Path\n\n\ndef f(start: Path | None):\n    return (start or Path.cwd()).resolve()\n"
    assert _flagged(tmp_path, source) == []


def test_a_name_annotated_as_something_else_elsewhere_is_still_flagged(tmp_path):
    source = "from pathlib import Path\n\n\n" "def g(start: Path):\n    return start\n\n\n" "def f(start: int):\n    return start or 5\n"
    assert _flagged(tmp_path, source) == [9]


def test_regex_patterns_tried_in_turn_are_not_a_trap(tmp_path):
    source = 'import re\n\n\ndef f(spec):\n    return re.search(r">=(\\d+)", spec) or re.search(r"~=(\\d+)", spec)\n'
    assert _flagged(tmp_path, source) == []


def test_two_different_numeric_fields_chained_with_or_stay_flagged(tmp_path):
    """A confirmed real bug class: a legitimate 0.0 cost replaced by the other field."""
    source = 'def f(bundle):\n    return bundle.get("effective_cost_usd") or bundle.get("actual_cost_usd")\n'
    assert _flagged(tmp_path, source) != []


def test_the_same_key_read_twice_is_still_flagged(tmp_path):
    source = 'def f(spec):\n    return spec.get("rev") or spec.get("rev", 3)\n'
    assert _flagged(tmp_path, source) != []
