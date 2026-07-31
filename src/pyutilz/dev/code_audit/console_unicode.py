"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- non-ASCII console/log output -----------------------------------------

_LOG_METHOD_NAMES = {"info", "warning", "error", "critical", "debug", "exception", "log"}


def _is_console_call(call: ast.Call) -> bool:
    """True for a ``print(...)`` call or a logging-method call (``logger.info`` etc.)."""
    func = call.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    if isinstance(func, ast.Attribute) and func.attr in _LOG_METHOD_NAMES:
        return True
    return False


def _first_str_arg_value(call: ast.Call) -> Optional[str]:
    """If the first positional arg is a string literal (plain or f-string), return it."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, ast.JoinedStr):
        parts = [v.value for v in first.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return "".join(parts) if parts else None
    return None


def _has_non_ascii(s: str) -> bool:
    """True if ``s`` contains any character outside the printable-ASCII range."""
    return any(ord(c) > 127 for c in s)


def scan_console_unicode(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``print(...)`` / ``logger.X(...)`` calls whose first string-literal argument contains a
    non-ASCII character.

    On Windows, default stdout/stderr encoding is ``cp1251``/``cp1252``; printing a fancy Unicode
    arrow, checkmark, or any non-Latin character crashes with ``UnicodeEncodeError`` -- silently
    fine on Linux/macOS dev machines, guaranteed broken on Windows.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_console_call(node):
                continue
            literal = _first_str_arg_value(node)
            if not literal or not _has_non_ascii(literal):
                continue
            findings.append(Finding(
                check="console_unicode",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    "print()/logger.*() call emits a non-ASCII string literal; crashes with "
                    "UnicodeEncodeError on Windows cp1251/cp1252 stdout. Replace fancy characters "
                    "(arrows, checkmarks, em-dash) with ASCII equivalents."
                ),
            ))
    return findings
