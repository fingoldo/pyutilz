"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- duplicate module docstring (second literal is dead code) -------------
#
# Only the FIRST top-level string-literal statement in a module becomes `__doc__`; a second one
# immediately after it is a no-op `ast.Expr` statement, parsed and silently discarded. Found
# independently by two different audit agents in matrix.py: the SECOND string was the more
# complete, accurate description ("...and memory-usage helpers...") -- the one actually shown by
# `help()`/mkdocs/IDE tooltips was the shorter, stale first one.


def scan_duplicate_module_docstring(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find modules whose first TWO top-level statements are both bare string-literal
    expressions -- only the first becomes ``__doc__``; the second is silently dead code.

    Severity: Low (docs/dead-code, not a runtime bug -- but the discarded string is often the
    more complete/accurate one, per the confirmed example above).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None or len(tree.body) < 2:
            continue
        first, second = tree.body[0], tree.body[1]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and isinstance(second, ast.Expr)
            and isinstance(second.value, ast.Constant)
            and isinstance(second.value.value, str)
        ):
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        findings.append(Finding(
            check="duplicate_module_docstring",
            severity="Low",
            file=rel,
            line=second.lineno,
            snippet=_line_text(src_lines, second.lineno),
            detail=(
                "Second top-level string literal right after the module docstring -- only the "
                "FIRST becomes `__doc__`; this one is a silently-discarded no-op expression "
                "statement. Delete one (or merge them)."
            ),
        ))
    return findings
