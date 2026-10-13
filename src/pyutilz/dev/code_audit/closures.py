"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

from ._base import walk_tree, Finding, _DEFAULT_EXCLUDE_DIRS, _arg_names, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- Class B: late-binding closures over loop vars ----------------------


def _names_referenced(tree: ast.AST) -> set[str]:
    """Set of ``Name(id=...)`` strings loaded inside ``tree`` (not stored)."""
    used: set[str] = set()
    for n in walk_tree(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
    return used


def _loop_target_names(node: ast.For | ast.AsyncFor) -> set[str]:
    """Names bound by a for-loop target (handles tuple unpacking) -- a ``For``/``AsyncFor``
    node's target is structurally the same shape ``_loop_target_names_from_target`` already
    walks for comprehensions, so this just unwraps ``.target`` and delegates."""
    return _loop_target_names_from_target(node.target)


_SYNCHRONOUS_CONSUMERS = frozenset({
    # builtins that exhaust their callable arg before returning.
    "sorted", "min", "max", "filter", "map", "any", "all", "sum", "next",
    "reduce", "groupby", "takewhile", "dropwhile",
})

_STORING_METHODS = frozenset({
    "append", "extend", "insert", "setdefault", "__setitem__",
    "register", "add_callback", "schedule", "submit", "apply_async",
})


def _ancestor_chain(target: ast.AST, root: ast.stmt) -> list[ast.AST]:
    """Return ancestors of ``target`` inside ``root``, root-first.

    Empty list if ``target`` is not within ``root``.
    """
    chain: list[ast.AST] = []
    found = [False]

    def _walk(node: ast.AST, stack: list[ast.AST]) -> None:
        """Depth-first search from ``node`` for ``target``; when found, records the accumulated ``stack`` of ancestors into the enclosing ``chain`` and sets ``found[0]``."""
        if found[0]:
            return
        if node is target:
            chain.extend(stack)
            found[0] = True
            return
        for child in ast.iter_child_nodes(node):
            _walk(child, [*stack, node])

    _walk(root, [])
    return chain


def _comprehension_target_names(node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> set[str]:
    """Names bound by any ``for`` clause of a comprehension/generator expression."""
    names: set[str] = set()
    for gen in node.generators:
        names |= _loop_target_names_from_target(gen.target)
    return names


def _loop_target_names_from_target(target: ast.AST) -> set[str]:
    """Same tuple/starred-unpacking logic as ``_loop_target_names``, but taking a bare target
    (used by comprehensions, whose ``ast.comprehension.target`` isn't wrapped in a For node)."""
    names: set[str] = set()

    def _walk(t: ast.AST) -> None:
        """Recursively collect Name ids bound by a (possibly nested/starred) comprehension target into the enclosing ``names`` set."""
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                _walk(el)
        elif isinstance(t, ast.Starred):
            _walk(t.value)

    _walk(target)
    return names


def _enclosing_call(chain: list) -> tuple[Optional[ast.Call], Optional[str]]:
    """Closest enclosing Call in ``chain`` (leaf-first walk) and the keyword name it passes the closure as, if any."""
    for idx in range(len(chain) - 1, -1, -1):
        node = chain[idx]
        if isinstance(node, ast.Call):
            kw_name: Optional[str] = None
            if idx + 1 < len(chain) and isinstance(chain[idx + 1], ast.keyword):
                kw_name = chain[idx + 1].arg
            return node, kw_name
    return None, None


def _call_escape_verdict(enclosing_call: ast.Call, kw_name: Optional[str]) -> bool:
    """Whether the closure passed into ``enclosing_call`` (as ``kw_name`` or positionally) is stored past the iteration."""
    func = enclosing_call.func
    func_name = None
    if isinstance(func, ast.Name):
        func_name = func.id
    elif isinstance(func, ast.Attribute):
        func_name = func.attr
    if kw_name == "key":
        return False  # ``key=lambda ...`` is exhausted by the call
    if func_name in _SYNCHRONOUS_CONSUMERS:
        return False
    if func_name in _STORING_METHODS:
        return True
    # functools.partial(lambda, ...) stores the closure.
    if func_name == "partial":
        return True
    # Default: a Call wrapping the lambda whose semantics we
    # don't know. Conservatively NOT an escape (avoid noise).
    return False


def _closure_escapes_iteration(closure_node: ast.AST, loop_body: list[ast.stmt]) -> bool:
    """Conservative check: does ``closure_node`` look like it escapes
    this iteration of the loop body?

    Walk up the ancestor chain from ``closure_node`` toward the
    enclosing statement. The closure escapes iff the FIRST enclosing
    Call passes it to a known storage method (``append``, ``register``,
    ``functools.partial``, etc.), OR there is no enclosing Call at all
    and the enclosing statement is an Assign / Return / Yield.

    Importantly: if the FIRST enclosing Call is a synchronous consumer
    (``sorted``, ``min``, ``max``, ``filter``, ``map``, ``functools.reduce``,
    or any callable passed as ``key=``), the closure is exhausted inside
    that call before iteration ends -> NOT an escape, regardless of how
    the call result is stored downstream.
    """
    for stmt in loop_body:
        chain = _ancestor_chain(closure_node, stmt)
        if not chain:
            continue
        enclosing_call, kw_name = _enclosing_call(chain)
        if enclosing_call is not None:
            return _call_escape_verdict(enclosing_call, kw_name)
        # No enclosing Call: look at the enclosing statement type.
        if isinstance(stmt, ast.Assign):
            return True
        if isinstance(stmt, (ast.Return, ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(stmt, ast.Expr):
            return False
        return False
    return False


def _loop_closure_findings(node: Any, loop_vars: set, rel: str, src_lines: list) -> list[Finding]:
    """Findings for lambdas and nested defs inside one ``for`` loop that capture its variables and escape the iteration."""
    found: list[Finding] = []
    for sub in ast.walk(node):
        if sub is node:
            continue
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found += _loop_def_finding(node, sub, loop_vars, rel, src_lines)
            continue
        if not isinstance(sub, ast.Lambda):
            continue
        own_defaults = {a.arg for a in (*sub.args.posonlyargs, *sub.args.args, *sub.args.kwonlyargs)}
        referenced = _names_referenced(sub.body)
        captured = (referenced & loop_vars) - own_defaults
        if not captured:
            continue
        if not _closure_escapes_iteration(sub, node.body):
            continue
        found.append(Finding(
            check="late_binding_closure",
            severity="P1",
            file=rel,
            line=sub.lineno,
            snippet=_line_text(src_lines, sub.lineno),
            detail=(
                f"lambda inside for-loop captures loop var(s) "
                f"{sorted(captured)!r}; closure escapes iteration "
                f"(stored/yielded). Bind as default: "
                f"`lambda x=x: ...`."
            ),
        ))
    return found


def _loop_def_finding(node: Any, sub: Any, loop_vars: set, rel: str, src_lines: list) -> list[Finding]:
    """Return a one-item list with the finding for a nested ``def`` capturing loop variables that escape, else []."""
    own_args = set(_arg_names(sub))
    referenced: set = set()
    for inner in sub.body:
        referenced |= _names_referenced(inner)
    captured = (referenced & loop_vars) - own_args
    if not captured:
        return []
    # Heuristic for escape: the nested def is decorated or assigned.
    if not (sub.decorator_list or _closure_escapes_iteration(sub, node.body)):
        return []
    return [
        Finding(
            check="late_binding_closure",
            severity="P1",
            file=rel,
            line=sub.lineno,
            snippet=_line_text(src_lines, sub.lineno),
            detail=(f"nested def {sub.name!r} inside for-loop " f"captures loop var(s) {sorted(captured)!r}; " f"closure escapes iteration."),
        )
    ]


def _comprehension_lambda_findings(node: Any, loop_vars: set, rel: str, src_lines: list) -> list[Finding]:
    """Findings for lambdas inside one comprehension or generator expression that capture its loop variables."""
    found: list[Finding] = []
    elts = [node.elt] if hasattr(node, "elt") else [node.key, node.value]
    for elt in elts:
        for sub in ast.walk(elt):
            if not isinstance(sub, ast.Lambda):
                continue
            own_defaults = {a.arg for a in (*sub.args.posonlyargs, *sub.args.args, *sub.args.kwonlyargs)}
            referenced = _names_referenced(sub.body)
            captured = (referenced & loop_vars) - own_defaults
            if not captured:
                continue
            # A comprehension/generator lambda's closure is, by construction, stored
            # into the resulting list/set/dict/generator -- it always escapes the
            # iteration that created it, unlike a bare `for` loop where escape depends
            # on what the loop body does with the closure.
            found.append(Finding(
                check="late_binding_closure",
                severity="P1",
                file=rel,
                line=sub.lineno,
                snippet=_line_text(src_lines, sub.lineno),
                detail=(
                    f"lambda inside comprehension/generator expression captures loop var(s) "
                    f"{sorted(captured)!r}; every lambda in the comprehension shares the same "
                    f"post-iteration binding. Bind as default: `lambda x=x: ...`."
                ),
            ))
    return found


def _file_closure_findings(tree: ast.AST, rel: str, src_lines: list) -> list[Finding]:
    """All late-binding closure findings for one parsed module."""
    findings: list[Finding] = []
    for node in walk_tree(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            loop_vars = _loop_target_names(node)
            if loop_vars:
                findings += _loop_closure_findings(node, loop_vars, rel, src_lines)
    for node in walk_tree(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            loop_vars = _comprehension_target_names(node)
            if loop_vars:
                findings += _comprehension_lambda_findings(node, loop_vars, rel, src_lines)
    return findings


def scan_late_binding_closures(root: Path,
                               exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                               ) -> list[Finding]:
    """Find ``lambda`` / nested ``def`` inside a ``for`` loop that
    references the loop variable AND escapes the iteration.

    Defaults shielding (``lambda x=x: ...``) is detected and excluded:
    if the closure has the loop var bound as its own positional
    default, the late-binding hazard is neutralized.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        findings += _file_closure_findings(tree, py.relative_to(root).as_posix(), _read_src_lines(py))
    return findings
