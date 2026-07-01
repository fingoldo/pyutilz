"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text


# --- mutation-during-iteration ------------------------------------------


_MUTATING_DICT_METHODS = frozenset({
    "pop", "popitem", "clear", "update", "setdefault",
})
_MUTATING_LIST_METHODS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear",
})
_MUTATING_SET_METHODS = frozenset({
    "add", "discard", "remove", "pop", "clear", "update",
    "difference_update", "intersection_update", "symmetric_difference_update",
})


def _iter_target_chain(node: ast.AST) -> Optional[str]:
    """Return a dotted-name chain for the iterated collection, or None
    for complex expressions (call results, comprehensions, etc.) that
    we can't statically associate with a mutation site.

    Examples:
        ``self._cache`` -> ``"self._cache"``
        ``d`` -> ``"d"``
        ``d.copy()`` -> None (a fresh object; mutation on `d` doesn't affect it)
        ``list(d)`` -> None (same; explicit defensive copy)
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _iter_target_chain(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    # Calls / subscripts: only ``.items() / .keys() / .values()`` calls on a
    # known chain are treated as live views; everything else is opaque.
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"items", "keys", "values"}:
            return _iter_target_chain(node.func.value)
    return None


def scan_mutation_during_iteration(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``for x in d: ... d[k] = v / del d[k] / d.append(...)`` shapes.

    Catches dict / list / set mutation while a live view (``d.keys()``,
    ``d.items()``, ``d.values()``, or the collection directly) is being
    iterated. Defensive-copy patterns (``for k in list(d):``,
    ``d.copy().items()``) are correctly NOT flagged because the iterated
    chain resolves to None.

    Severity: P0. Dict / set mutation raises RuntimeError loudly at the
    next iteration; list mutation silently skips elements (correctness
    bug). The audit from wave 32 (2026-05-20) confirmed this is a
    high-leverage scanner -- mlframe came back clean but the scanner
    pins the discipline for future contributors.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor)):
                continue
            iter_chain = _iter_target_chain(node.iter)
            if iter_chain is None:
                continue  # Iterated chain wasn't a live view we can track.
            for child in ast.walk(node):
                if child is node:
                    continue
                # del d[...]
                if isinstance(child, ast.Delete):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Subscript):
                            sub_chain = _iter_target_chain(tgt.value)
                            if sub_chain == iter_chain:
                                findings.append(Finding(
                                    check="mutation_during_iteration",
                                    severity="P0",
                                    file=rel,
                                    line=child.lineno,
                                    snippet=_line_text(src_lines, child.lineno),
                                    detail=(
                                        f"``del {iter_chain}[...]`` while iterating "
                                        f"the same collection at line {node.lineno}. "
                                        f"Dict / set mutation raises RuntimeError; "
                                        f"list mutation silently skips elements. "
                                        f"Iterate over ``list({iter_chain})`` or "
                                        f"``{iter_chain}.copy()`` instead."
                                    ),
                                ))
                # d[k] = v  -- only when adding a new key (size change)
                # The safe case is reassignment of an existing key, which
                # is size-preserving and not raised by CPython. We can't
                # statically tell new vs existing, so flag conservatively
                # at severity P1 (one rung lower than del / .pop).
                if isinstance(child, ast.Assign):
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Subscript):
                            sub_chain = _iter_target_chain(tgt.value)
                            if sub_chain == iter_chain:
                                findings.append(Finding(
                                    check="mutation_during_iteration",
                                    severity="P1",
                                    file=rel,
                                    line=child.lineno,
                                    snippet=_line_text(src_lines, child.lineno),
                                    detail=(
                                        f"``{iter_chain}[...] = ...`` while iterating "
                                        f"the same collection at line {node.lineno}. "
                                        f"Reassigning an EXISTING key is size-preserving "
                                        f"and safe (CPython); ADDING a new key changes "
                                        f"size and raises RuntimeError on dict / set. "
                                        f"Statically indistinguishable -- inspect."
                                    ),
                                ))
                # d.append(...) / d.pop(...) / etc.
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    method = child.func.attr
                    receiver_chain = _iter_target_chain(child.func.value)
                    if receiver_chain == iter_chain and method in (
                        _MUTATING_DICT_METHODS
                        | _MUTATING_LIST_METHODS
                        | _MUTATING_SET_METHODS
                    ):
                        findings.append(Finding(
                            check="mutation_during_iteration",
                            severity="P0",
                            file=rel,
                            line=child.lineno,
                            snippet=_line_text(src_lines, child.lineno),
                            detail=(
                                f"``{iter_chain}.{method}(...)`` while iterating "
                                f"the same collection at line {node.lineno}. "
                                f"Iterate over a copy: ``for x in list({iter_chain}):`` "
                                f"or ``{iter_chain}.copy()``."
                            ),
                        ))
    return findings
