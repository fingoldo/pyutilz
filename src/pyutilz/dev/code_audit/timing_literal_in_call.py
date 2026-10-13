"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import walk_tree, Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- a bare numeric literal that sets a wait, timeout or backoff at its call site ---------------
#
# `_facade.sleep(10 * _facade.random())` tuned a retry backoff with a 10 nobody could find or change.
# ruff does not catch it: PLR2004 (magic-value-comparison) only looks at comparisons, and the arithmetic
# here is not one. A wait that matters is a named parameter with a default, or a named module constant,
# so a caller can see it and a test can change it.
#
# Scope is deliberately narrow: literals inside a sleep call, or the value of a keyword that names a
# backoff or jitter. Not flagged: HTTP timeouts (a `timeout=15` on a request is a reviewed choice, and
# there are hundreds), psutil sampling intervals, tests (short sleeps there are intentional), 0, and
# literals assigned to a name or used as a parameter default.

_SLEEP_CALLS = frozenset({"sleep", "sleep_async"})
_BACKOFF_KEYWORDS = frozenset({"backoff", "jitter", "retry_wait", "retry_delay"})
_SKIPPED_TOP_DIRS = frozenset({"tests", "scripts"})


def _call_name(call: ast.Call) -> str | None:
    """The simple name of the called function, for ``f(...)`` and ``obj.f(...)`` alike."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _nonzero_numbers(node: ast.AST) -> list[ast.Constant]:
    """Every numeric constant under ``node`` other than 0 and booleans, and not an index offset.

    A literal that is a direct operand of ``+`` or ``-`` is index arithmetic (``base * (attempt + 1)``),
    not a duration, so it is skipped.
    """
    offsets = {
        id(operand)
        for parent in ast.walk(node)
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub))
        for operand in (parent.left, parent.right)
    }
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, (int, float))
        and not isinstance(child.value, bool)
        and child.value != 0
        and id(child) not in offsets
    ]


def _timing_literals_in_calls(tree: ast.AST) -> list[ast.Constant]:
    """The duration literals found in sleep/wait arguments and duration-named keyword values."""
    hits: list[ast.Constant] = []
    for node in walk_tree(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) in _SLEEP_CALLS:
            for arg in node.args:
                hits.extend(_nonzero_numbers(arg))
        for kw in node.keywords:
            if kw.arg in _BACKOFF_KEYWORDS:
                hits.extend(_nonzero_numbers(kw.value))
    return hits


def scan_timing_literal_in_call(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Flag bare numeric literals that set a wait, timeout or backoff at the call site.

    Severity Low: the code runs, but the duration cannot be tuned or tested without editing the body.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        if rel.split("/", 1)[0] in _SKIPPED_TOP_DIRS:
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        findings.extend(
            Finding(
                check="timing_literal_in_call",
                severity="Low",
                file=rel,
                line=const.lineno,
                snippet=_line_text(src_lines, const.lineno),
                detail=(
                    f"bare duration literal {const.value!r} sets a wait or timeout at its call site; "
                    "make it a named parameter with a default, or a named module constant, so callers and tests can change it."
                ),
            )
            for const in _timing_literals_in_calls(tree)
        )
    return findings
