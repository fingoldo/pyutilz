"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# Call names that release a pooled/held resource back for reuse. Projects with
# their own release-function naming convention pass their own set via the
# ``release_call_names`` kwarg.
DEFAULT_RELEASE_CALL_NAMES: frozenset[str] = frozenset({
    "release_conn", "_release_conn", "release_connection", "putconn", "release_singleton_lock",
})


def _names_read_in(node: ast.AST) -> set[str]:
    """Every identifier read (``ast.Load`` context) anywhere inside ``node``."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _release_call_args_in_finally_blocks(func_node: ast.AST, exclude_subtree: ast.AST, release_call_names: frozenset[str]) -> set[str]:
    """Names passed as the FIRST argument to a release-shaped call inside any
    ``finally:`` block within ``func_node``, excluding anything inside
    ``exclude_subtree`` (the shielded closure's own body -- its own finally
    releasing its OWN resource is exactly the correct, safe pattern).

    Only the first argument -- by convention across release-shaped calls
    (``_release_conn(conn, sem_ok, semaphore)``, ``release_connection(conn)``,
    ``putconn(conn)``), that's the actual exclusively-owned resource being
    handed back. Later arguments (a bool flag, an ``asyncio.Semaphore``) are
    safe to reference from both the outer scope and the shielded closure -- a
    semaphore is specifically designed for concurrent acquire/release from
    multiple tasks, unlike a raw connection object.
    """
    exclude_nodes = set(ast.walk(exclude_subtree))
    found: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Try) or node in exclude_nodes:
            continue
        for stmt in node.finalbody:
            for call in ast.walk(stmt):
                if call in exclude_nodes or not isinstance(call, ast.Call):
                    continue
                func = call.func
                name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
                if name in release_call_names and call.args and isinstance(call.args[0], ast.Name):
                    found.add(call.args[0].id)
    return found


def scan_shielded_resource_release_race(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    release_call_names: frozenset[str] = DEFAULT_RELEASE_CALL_NAMES,
) -> list[Finding]:
    """Find ``asyncio.shield(closure())`` where the shielded closure reads a
    name that the ENCLOSING function's own ``finally:`` also hands to a
    release-shaped call, outside the shielded closure.

    ``asyncio.shield`` protects the shielded TASK from the caller's own
    cancellation, not the caller's local variables -- if the caller's
    ``finally:`` releases a pooled resource (a DB connection, a lock) back
    for reuse while the shielded task may still be mid-write on that exact
    resource, a concurrently-acquiring caller races onto the same object.
    The fix is for the shielded closure to acquire and release its OWN,
    fully independent resource in its own ``finally:``, never sharing the
    outer function's.

    This is deliberately narrow: it only flags the CONCRETE shape (shielded
    closure reads name X; enclosing finally: releases X outside that
    closure), not "any asyncio.shield use" -- whether a release genuinely
    races a shielded task depends on understanding what the resource is and
    who else can acquire it, which this scanner can't determine; it only
    catches the exact reintroduction of an already-diagnosed bug shape.

    Severity: P0 (silent cross-request data corruption on a pooled
    connection/resource, not an immediate crash).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for outer in ast.walk(tree):
            if not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nested_async_defs = {n.name: n for n in ast.walk(outer) if isinstance(n, ast.AsyncFunctionDef) and n is not outer}
            if not nested_async_defs:
                continue
            for node in ast.walk(outer):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shield"):
                    continue
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "asyncio"):
                    continue
                if not node.args or not isinstance(node.args[0], ast.Call):
                    continue
                shielded_call = node.args[0]
                shielded_func = shielded_call.func
                closure_name = shielded_func.id if isinstance(shielded_func, ast.Name) else None
                if closure_name not in nested_async_defs:
                    continue
                closure_node = nested_async_defs[closure_name]
                closure_reads = _names_read_in(closure_node)
                released_elsewhere = _release_call_args_in_finally_blocks(outer, closure_node, release_call_names)
                overlap = closure_reads & released_elsewhere
                if not overlap:
                    continue
                findings.append(Finding(
                    check="shielded_resource_release_race",
                    severity="P0",
                    file=rel,
                    line=outer.lineno,
                    snippet=_line_text(src_lines, outer.lineno),
                    detail=(
                        f"asyncio.shield({closure_name}()) reads {sorted(overlap)}, also released in an "
                        "enclosing finally: block outside the shielded closure -- the shielded task can "
                        "still be using the resource the caller's own finally: just released back to the "
                        "pool. Acquire a fully independent resource INSIDE the shielded closure and "
                        "release it in that closure's own finally:, never sharing the outer function's."
                    ),
                ))
    return findings
