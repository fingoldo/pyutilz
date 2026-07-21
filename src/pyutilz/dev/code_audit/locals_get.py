"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text


def _is_locals_or_globals_call(node: ast.AST) -> bool:
    """True for a bare, no-argument ``locals()``/``globals()`` call node."""
    return isinstance(node, ast.Call) and not node.args and not node.keywords and isinstance(node.func, ast.Name) and node.func.id in ("locals", "globals")


def scan_locals_get_fragile_lookup(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``locals().get(name, default)`` / ``globals().get(name, default)``
    used to read a conditionally-assigned local/global instead of a normal,
    pre-initialized variable.

    ``locals()``/``globals()`` return a snapshot dict at call time -- reading
    through ``.get(...)`` works today only because the name happens to
    already be bound in that snapshot. A refactor that renames the variable,
    moves the read to before the conditional assignment, or extracts the
    surrounding code into a helper (where ``locals()`` now snapshots a
    DIFFERENT, smaller scope) silently changes this from "read the real
    value" to "always get the default" -- no error, no warning, just a wrong
    value. A plain variable, initialized once (``x = default`` before the
    conditional, then reassigned inside it), doesn't have this fragility:
    a rename/reorder/extraction breaks it loudly (``NameError``/``UnboundLocalError``)
    instead of silently.

    Severity: P1 (silent wrong-value risk on a routine refactor, not an
    immediate bug -- the lookup is correct as written today).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
                continue
            if not _is_locals_or_globals_call(node.func.value):
                continue
            scope_name = node.func.value.func.id  # type: ignore[attr-defined]  # narrowed by _is_locals_or_globals_call above
            findings.append(Finding(
                check="locals_get_fragile_lookup",
                severity="P1",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"{scope_name}().get(...) reads a conditionally-assigned name via a "
                    f"scope-snapshot dict instead of a normal variable -- a rename, "
                    f"reorder, or extraction-into-a-helper silently changes this from "
                    f"'read the real value' to 'always get the default', with no error. "
                    f"Use a normally-initialized variable instead."
                ),
            ))
    return findings
