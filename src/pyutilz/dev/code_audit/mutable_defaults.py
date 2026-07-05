"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- Class A: mutable defaults ------------------------------------------


_MUTATING_METHODS = frozenset({
    "append", "extend", "insert", "pop", "clear", "remove", "sort",
    "reverse", "setdefault", "update", "add", "discard", "popitem",
    "__setitem__", "__delitem__", "__iadd__", "__ior__",
})

_MUTABLE_LITERAL_TYPES = (ast.List, ast.Dict, ast.Set)
_MUTABLE_CALLS = frozenset({"list", "dict", "set"})


def _is_mutable_default(default: ast.AST) -> Optional[str]:
    """Return a short label if ``default`` is a mutable literal/call,
    else None. ``label`` is one of ``"list"``, ``"dict"``, ``"set"``."""
    if isinstance(default, ast.List):
        return "list"
    if isinstance(default, ast.Dict):
        return "dict"
    if isinstance(default, ast.Set):
        return "set"
    if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
        if default.func.id in _MUTABLE_CALLS and not default.args and not default.keywords:
            return default.func.id
    return None


def _param_is_mutated(func: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> bool:
    """Heuristic: does the function body apply a mutating operation to
    the named parameter? Walks calls of shape ``param.append(...)``,
    subscript-assigns ``param[k] = v``, augmented assigns ``param += x``."""
    for node in ast.walk(func):
        # param.<mutating_method>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func
            if isinstance(attr.value, ast.Name) and attr.value.id == param_name and attr.attr in _MUTATING_METHODS:
                return True
        # param[k] = v  /  del param[k]
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == param_name:
                    return True
                if isinstance(t, ast.Name) and t.id == param_name and isinstance(node, ast.AugAssign):
                    return True
    return False


def scan_mutable_defaults(root: Path,
                          exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                          ) -> list[Finding]:
    """Find ``def f(x=[]) / ={} / =set() / =list() / =dict() / =set()`` defaults.

    Severity rule: parameter mutated in body -> P0 (state leaks across
    callers); not mutated -> Low (idiomatic-but-questionable, no
    observable bug today but reader trap and future-bug bait).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            # Pair every positional default with its corresponding name.
            pos_args = (*args.posonlyargs, *args.args)
            n_defaults = len(args.defaults)
            if n_defaults:
                paired = list(zip(pos_args[-n_defaults:], args.defaults))
            else:
                paired = []
            paired.extend((k, d) for k, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None)
            for arg_node, default in paired:
                label = _is_mutable_default(default)
                if label is None:
                    continue
                mutated = _param_is_mutated(node, arg_node.arg)
                sev = "P0" if mutated else "Low"
                detail = f"def {node.name}(..., {arg_node.arg}={label}()): " + (
                    "MUTATED in body -> shared state leaks across callers" if mutated else "default is mutable but never mutated in body (Low)"
                )
                findings.append(Finding(
                    check="mutable_default",
                    severity=sev,
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=detail,
                ))
    return findings
