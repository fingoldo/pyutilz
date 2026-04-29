"""H1 — meta-test that ``logger.info(...)`` / ``logger.debug(...)`` etc.
calls don't pre-format their messages with f-strings or ``%`` /
``.format()``.

The standard library's ``logging`` API supports lazy-formatting via::

    logger.info("processed %d rows in %.2fs", n, elapsed)

The format string + args are only stitched together when the log
record actually gets emitted — which lets disabled levels skip the
work entirely. f-string-style formatting::

    logger.info(f"processed {n} rows in {elapsed:.2f}s")

forces the format every call site, even when the level is disabled.

For DEBUG / INFO calls in hot paths, this measurably slows things down
(repeated millions of times per training run, each call building a
string that's then thrown away). It's also harder to scrub from
structured-log pipelines because the formatted-already string can't
be re-templated.

Snapshot-style. Baseline locks current state; new f-string log calls
fail unless added to baseline via ``--refresh-logger-baseline``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_logger_lazy_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")

# Methods we audit. WARNING / ERROR / CRITICAL paths are exempt because
# (a) they're rare relative to INFO/DEBUG, (b) f-string is more
# readable when the message is genuinely going to be emitted.
_AUDITED_METHODS = {"debug", "info"}


def _refresh_requested() -> bool:
    return "--refresh-logger-baseline" in sys.argv


def _is_logger_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _AUDITED_METHODS:
        return False
    # Must be ``X.<method>`` where X is some Name (typically ``logger``
    # or ``logging``) — heuristic, but typical-pyutilz calls look this way.
    return isinstance(func.value, (ast.Name, ast.Attribute))


def _first_arg_is_eager_format(call: ast.Call) -> bool:
    if not call.args:
        return False
    first = call.args[0]
    # f-string
    if isinstance(first, ast.JoinedStr):
        # An f-string with at least one FormattedValue — eager format.
        return any(isinstance(v, ast.FormattedValue) for v in first.values)
    # ``"... %s ..." % (...)`` — BinOp with Mod
    if isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mod):
        return True
    # ``"...".format(...)`` — Call where func is Attribute(attr="format")
    if isinstance(first, ast.Call):
        f = first.func
        if isinstance(f, ast.Attribute) and f.attr == "format":
            return True
    # ``"a" + var`` style concat (less common, but seen)
    if isinstance(first, ast.BinOp) and isinstance(first.op, ast.Add):
        return True
    return False


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
            if not _is_logger_call(node):
                continue
            if _first_arg_is_eager_format(node):
                out.add(f"{rel}:{node.lineno}")
    return out


def test_no_new_eager_log_format_on_debug_or_info():
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2), encoding="utf-8"
        )
        pytest.skip(
            f"logger lazy-format baseline refreshed at "
            f"{_BASELINE_PATH.name} ({len(current)} eager call site(s))"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_eager_log_format_on_debug_or_info] "
            f"{len(fixed)} call site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh baseline to lock in.\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new ``logger.debug/info`` call(s) using eager "
            f"formatting (f-string / %-format / .format() / +-concat). "
            f"Replace with lazy form ``logger.info(\"...%s...\", val)`` "
            f"so disabled log levels can skip the format work, OR "
            f"refresh the baseline if intentional:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
