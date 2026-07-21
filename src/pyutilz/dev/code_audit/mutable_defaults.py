"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text, _arg_names

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


def _shadows_param(node: ast.AST, param_name: str) -> bool:
    """True if a nested function/lambda redeclares ``param_name`` as its own parameter,
    shadowing the outer binding for its entire body."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return False
    args = node.args
    all_names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        all_names.add(args.vararg.arg)
    if args.kwarg is not None:
        all_names.add(args.kwarg.arg)
    return param_name in all_names


def _walk_skipping_shadowed_scopes(func: ast.AST, param_name: str) -> Iterator[ast.AST]:
    """Like ast.walk(func), but stops descending into nested function/lambda scopes that
    redeclare param_name as their own parameter (a shadow of the outer binding)."""
    todo = list(ast.iter_child_nodes(func))
    while todo:
        node = todo.pop()
        yield node
        if _shadows_param(node, param_name):
            continue
        todo.extend(ast.iter_child_nodes(node))


def _param_is_mutated(func: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> bool:
    """Heuristic: does the function body apply a mutating operation to
    the named parameter? Walks calls of shape ``param.append(...)``,
    subscript-assigns ``param[k] = v``, augmented assigns ``param += x``.
    Does not descend into nested functions/lambdas that shadow ``param_name`` with their
    own parameter of the same name -- those mutate an independently-scoped binding."""
    for node in _walk_skipping_shadowed_scopes(func, param_name):
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


def _find_param_aliasing_mutation(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str, int]]:
    """Find ``local = param`` (bare Name-to-Name rebind, no ``.copy()``/``list()``/``dict()``/
    ``[*...]`` wrapping) followed later in the SAME function by an in-place mutation of ``local``
    (``+=``, ``.append``/``.extend``/``.update``/etc., or subscript-assign) -- the mutation then
    silently reaches the CALLER's object through ``param``, since no copy was ever made.

    Returns a list of ``(param_name, local_name, mutation_lineno)`` for each such site found.

    Nodes are processed in SOURCE-LINE order (not ``ast.walk``'s breadth-first order): mutually
    exclusive ``if``/``elif``/``else`` branches routinely reassign the same local name (e.g.
    ``res = obj`` in one branch, ``res = {}`` in a sibling branch, then ``res[key] = ...`` inside
    that sibling branch's own body) -- BFS order can pair a mutation from one branch against an
    alias established in a DIFFERENT, mutually-exclusive branch, a false positive found in the
    wild (``text/strings/jsonutils.py``) during the first real run of this scanner.
    """
    param_names = set(_arg_names(func))
    aliases: dict[str, str] = {}  # local_name -> param_name
    hits: list[tuple[str, str, int]] = []
    nodes = sorted(ast.walk(func), key=lambda n: getattr(n, "lineno", 0))
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            local_name = node.targets[0].id
            if isinstance(node.value, ast.Name) and node.value.id in param_names:
                aliases[local_name] = node.value.id
                continue
            elif local_name in aliases:
                del aliases[local_name]  # reassigned to something else -- no longer a bare alias
                continue
        if not aliases:
            continue
        target_name = None
        lineno = 0
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in aliases and node.func.attr in _MUTATING_METHODS:
                target_name = node.func.value.id
                lineno = node.lineno
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id in aliases:
            target_name = node.target.id
            lineno = node.lineno
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in aliases:
                    target_name = t.value.id
                    lineno = node.lineno
        if target_name is not None:
            hits.append((aliases[target_name], target_name, lineno))
    return hits


def scan_parameter_aliasing_mutation(root: Path,
                                     exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                                     ) -> list[Finding]:
    """Find ``local = param`` (bare rebind, no ``.copy()``) followed by an in-place mutation of
    ``local`` -- the mutation silently reaches the caller's object through the un-copied
    parameter reference.

    Confirmed real bug in the 2026-07-21 audit: ``database/db/upsert.py``'s
    ``returning_fields = history_fields; returning_fields += [hash_field]`` used
    ``list.__iadd__`` (in-place) to mutate the CALLER's own ``history_fields`` list, leaking
    ``hash_field`` into a history-table column list the caller never asked for it in.

    Severity: P0 (silent cross-call state corruption, same class as a mutable-default leak).
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
            for param_name, local_name, lineno in _find_param_aliasing_mutation(node):
                findings.append(Finding(
                    check="parameter_aliasing_mutation",
                    severity="P0",
                    file=rel,
                    line=lineno,
                    snippet=_line_text(src_lines, lineno),
                    detail=(
                        f"`{local_name} = {param_name}` (bare alias, no `.copy()`) then mutated "
                        f"in place -- this silently mutates the CALLER's `{param_name}` object "
                        "too, since no copy was ever made."
                    ),
                ))
    return findings


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
