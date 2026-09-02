"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a failure stored in the cache as though it were the answer --------------------------------
#
# The lookup fails, the handler writes the failure value into the cache, and from then on every
# caller is served that failure without the lookup ever being retried. A transient timeout becomes
# a permanent empty answer for the lifetime of the process -- and the longer the cache lives, the
# worse it gets.
#
# The audited codebase had this exact shape: a network probe that returned `None` on error, cached
# by key, so one blip pinned that key to `None` for the whole run. Nothing looked wrong: the cache
# was working perfectly, at storing the wrong thing.
#
# The signature is narrow on purpose:
#
# * the write happens inside an `except` handler, or in the `else`/fallback arm right after one --
#   somewhere the code already knows it is on the failure path.
# * the value written is a failure sentinel: `None`, or an empty container literal.
# * the target looks like a cache: a subscript assignment onto a name containing "cache"/"memo"/
#   "seen"/"store", or a `.set`/`.put`/`.setdefault` call on one.
#
# A negative cache is a real and deliberate technique, which is why the message says to make it
# explicit -- a TTL, or a distinct sentinel a reader can tell from a genuine empty result -- rather
# than claiming the write is always wrong.

_CACHE_HINTS = ("cache", "memo", "store", "seen", "lookup", "registry")
_WRITE_METHODS = frozenset({"set", "put", "setdefault", "store", "save"})


def _is_failure_sentinel(node: ast.expr) -> str | None:
    """The sentinel this expression is, if it is one: `None` or an empty container literal."""
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, (ast.List, ast.Tuple)) and not node.elts:
        return "an empty list" if isinstance(node, ast.List) else "an empty tuple"
    if isinstance(node, ast.Dict) and not node.keys:
        return "an empty dict"
    if isinstance(node, ast.Set) and not node.elts:
        return "an empty set"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "dict", "set", "tuple"} and not node.args:
        return f"an empty {node.func.id}"
    return None


def _cache_name(node: ast.expr) -> str | None:
    """The cache-looking name this expression writes to, if it looks like one."""
    if isinstance(node, ast.Name):
        text = node.id
    elif isinstance(node, ast.Attribute):
        text = node.attr
    else:
        return None
    return text if any(hint in text.lower() for hint in _CACHE_HINTS) else None


def _cache_write(stmt: ast.stmt) -> tuple[str, str] | None:
    """(cache name, sentinel) if this statement stores a failure sentinel into a cache."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Subscript):
            name = _cache_name(target.value)
            sentinel = _is_failure_sentinel(stmt.value)
            if name and sentinel:
                return name, sentinel
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr in _WRITE_METHODS and len(call.args) == 2:
            name = _cache_name(func.value)
            sentinel = _is_failure_sentinel(call.args[1])
            if name and sentinel:
                return name, sentinel
    return None


def scan_sentinel_cached_as_answer(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a failure sentinel written into a cache from inside an exception handler.

    The lookup fails, the handler caches the failure, and every later caller is served it without
    the lookup ever being retried. One transient timeout then pins that key to `None` for the
    lifetime of the process, and nothing looks wrong -- the cache works perfectly, at storing the
    wrong thing.

    A negative cache is a legitimate technique, so this asks for it to be made explicit (a TTL, or
    a sentinel a reader can tell apart from a genuine empty result) rather than calling every such
    write a defect.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            for stmt in ast.walk(handler):
                if not isinstance(stmt, ast.stmt):
                    continue
                write = _cache_write(stmt)
                if write is None:
                    continue
                name, sentinel = write
                findings.append(
                    Finding(
                        check="sentinel_cached_as_answer",
                        severity="P1",
                        file=rel,
                        line=stmt.lineno,
                        snippet=_line_text(src_lines, stmt.lineno),
                        detail=(
                            f"the handler at line {handler.lineno} writes {sentinel} into "
                            f"`{name}`, so a failed lookup is cached as though it were the answer. "
                            "Every later caller is served that failure and the lookup is never "
                            "retried -- one transient error pins the key for the process's "
                            "lifetime. If a negative cache is intended, give it a TTL or a "
                            "sentinel a reader can tell apart from a genuine empty result."
                        ),
                    )
                )
    return findings
