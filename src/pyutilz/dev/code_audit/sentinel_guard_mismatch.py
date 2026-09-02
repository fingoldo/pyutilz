"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a falsy in-band sentinel meeting a caller guard that tests `is None` ----------------------
#
# A function signals "unknown" or "failed" by returning `0`, `""`, `[]` or `-1` from an except
# handler, while its callers guard with `if X is None`. The sentinel satisfies neither branch of
# that guard, so the failure is indistinguishable from a legitimate value and the guard never
# fires.
#
# Confirmed instances in one audited codebase:
#
# * `wayback_cdx.py` returned `0` for a malformed numPages body. The caller guarded
#   `pages_total is None or (pages_total > 0 and ...)`, which `0` satisfies neither clause of --
#   so one transient HTML error page retired a discovery source PERMANENTLY.
# * A due-count probe returned `0` on exception, turning its own failure into "nothing is due".
#   The accepted fix was, verbatim, to make `None` a third answer.
#
# The same codebase also records two DELIBERATE choices in this space -- `get_count` returning
# `-1` rather than `0`, and a query returning `None` on error against `{}` on empty. Three separate
# occasions of having to reason about the distinction is the argument for checking it mechanically.

_FALSY_SENTINELS: tuple[object, ...] = (0, 0.0, "", -1)


def _returns_falsy_from_except(func: ast.AST) -> list[tuple[object, int]]:
    """Falsy constants returned from inside an `except` handler, with their lines.

    Restricted to except handlers on purpose: a function that returns `0` on the ordinary path is
    returning a number, not signalling anything. It is the FAILURE path returning a value
    indistinguishable from success that this rule is about.
    """
    out: list[tuple[object, int]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.ExceptHandler):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Return) or sub.value is None:
                continue
            value = sub.value
            if isinstance(value, ast.Constant) and any(value.value is s or (type(value.value) is type(s) and value.value == s) for s in _FALSY_SENTINELS):
                out.append((value.value, sub.lineno))
            elif isinstance(value, (ast.List, ast.Dict, ast.Tuple)) and not getattr(value, "elts", getattr(value, "keys", [])):
                out.append(("empty container", sub.lineno))
    return out


def _returns_none_anywhere(func: ast.AST) -> bool:
    """Does this function also return None explicitly? Then `is None` guards are legitimate."""
    for node in ast.walk(func):
        if isinstance(node, ast.Return):
            if node.value is None:
                return True
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                return True
    return False


def _is_none_guards(tree: ast.AST) -> dict[str, int]:
    """{name: line} for every `X is None` / `X is not None` guard, keyed by the name tested."""
    guards: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Constant) and right.value is None):
            continue
        left = node.left
        name = None
        if isinstance(left, ast.Name):
            name = left.id
        elif isinstance(left, ast.Attribute):
            name = left.attr
        if name:
            guards.setdefault(name, node.lineno)
    return guards


def _assigned_from_call(tree: ast.AST) -> dict[str, set[str]]:
    """{callee name: names it was assigned to} -- `pages = fetch_pages()` -> {"fetch_pages": {"pages"}}."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
        if not name:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.setdefault(name, set()).add(target.id)
            elif isinstance(target, ast.Attribute):
                out.setdefault(name, set()).add(target.attr)
    return out


def scan_sentinel_guard_mismatch(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find failure sentinels a caller's `is None` guard cannot see.

    A function returns `0`/`""`/`[]`/`-1` from an except handler; a caller assigns that result and
    guards it with `is None`. The sentinel passes the guard, so the failure reads as a legitimate
    value -- which is how a transient error page permanently retired a discovery source.

    Not reported when the function ALSO returns `None` somewhere: then `is None` is a guard against
    a real outcome and the falsy return is a second, deliberate answer. That is the shape the
    audited codebase chose twice on purpose.
    """
    findings: list[Finding] = []

    sentinel_returners: dict[str, tuple[str, object, int]] = {}
    parsed: list[tuple[Path, ast.Module, str]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        parsed.append((py, tree, rel))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _returns_none_anywhere(func):
                continue
            sentinels = _returns_falsy_from_except(func)
            if sentinels:
                value, line = sentinels[0]
                sentinel_returners[func.name] = (rel, value, line)

    if not sentinel_returners:
        return findings

    for py, tree, rel in parsed:
        guards = _is_none_guards(tree)
        if not guards:
            continue
        assigned = _assigned_from_call(tree)
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()

        for callee, names in sorted(assigned.items()):
            if callee not in sentinel_returners:
                continue
            guarded = sorted(names & set(guards))
            if not guarded:
                continue
            name = guarded[0]
            where, value, ret_line = sentinel_returners[callee]
            findings.append(
                Finding(
                    check="sentinel_guard_mismatch",
                    severity="P1",
                    file=rel,
                    line=guards[name],
                    snippet=_line_text(src_lines, guards[name]),
                    detail=(
                        f"`{name}` comes from `{callee}`, which returns {value!r} from an except "
                        f"handler ({where}:{ret_line}) and never returns None -- so this `is None` "
                        "guard cannot see the failure, and it reads as a legitimate value. Return "
                        "None for the failure, or test for the sentinel the function actually uses."
                    ),
                )
            )
    return findings
