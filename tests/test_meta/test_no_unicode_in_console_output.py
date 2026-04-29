"""E5 — meta-test that no production code emits non-ASCII characters
through ``print(...)`` / ``logger.info`` / ``logger.warning`` / etc.

On Windows, default stdout/stderr encoding is ``cp1251`` (Russian) or
``cp1252`` (Western European). Printing a fancy Unicode arrow ``→``,
emoji ``✓``, or any non-Latin character crashes with
``UnicodeEncodeError`` — silently fine on Linux / macOS dev machines,
guaranteed broken on Windows production. Documented in the maintainer's
own ``feedback_windows_encoding`` rule.

This test walks the AST of every production .py and flags string
constants passed as the first arg to a ``print`` / ``log*`` /
``logger.X`` call when the string contains characters outside ASCII.

Snapshot-style: existing offenders captured in a baseline (so this
landing doesn't immediately fail on legacy code); future commits that
add a new offender fail.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_console_unicode_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__",)

# Console-output sinks we audit. Cover bare ``print`` and the standard
# ``logger.X`` family.
_LOG_METHOD_NAMES = {"info", "warning", "error", "critical", "debug",
                     "exception", "log"}


def _refresh_requested() -> bool:
    return "--refresh-console-unicode-baseline" in sys.argv


def _is_console_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id in {"print"}:
        return True
    if isinstance(func, ast.Attribute):
        if func.attr in _LOG_METHOD_NAMES:
            return True
    return False


def _first_str_arg_value(call: ast.Call) -> str | None:
    """If the first positional arg is a string literal, return it."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    # f-strings are ``ast.JoinedStr`` — flatten the literal parts.
    if isinstance(first, ast.JoinedStr):
        parts: list[str] = []
        for v in first.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        return "".join(parts) if parts else None
    return None


def _has_non_ascii(s: str) -> bool:
    return any(ord(c) > 127 for c in s)


def _build_offending_set() -> set[str]:
    out: set[str] = set()
    for py in PYUTILZ_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_console_call(node):
                continue
            literal = _first_str_arg_value(node)
            if literal and _has_non_ascii(literal):
                out.add(f"{rel}:{node.lineno}")
    return out


def test_no_new_non_ascii_console_output():
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2),
            encoding="utf-8",
        )
        pytest.skip(
            f"console-Unicode baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} call site(s) with non-ASCII string literal)"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_non_ascii_console_output] "
            f"{len(fixed)} call site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh baseline to lock in.\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new ``print`` / ``logger.*`` call site(s) emit "
            f"non-ASCII string literals. On Windows this crashes with "
            f"UnicodeEncodeError on cp1251/cp1252 stdout. Replace fancy "
            f"chars (→, ✓, ✗, em-dash) with ASCII (->, [OK], [X], --), "
            f"OR refresh the baseline if intentional:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
