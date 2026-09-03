"""Scanner tests for dead_wiring, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_dead_public_callables,
)

from ._helpers import _write

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
