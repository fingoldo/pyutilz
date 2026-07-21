"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- discarded coroutine (missed await) ---------------------------------------
#
# ``do_save(item)`` as a bare statement, where ``do_save`` is an ``async def``
# in the same module, creates a coroutine object and immediately discards it:
# the function body NEVER RUNS. Python only surfaces this as a
# "coroutine ... was never awaited" RuntimeWarning -- easy to miss in logs,
# invisible in tests that don't check warnings -- while the caller carries on
# as if the save/notify/cleanup happened.
#
# Precision comes from three deliberate restrictions, each added after a
# same-shape false positive during 2026-07 corpus validation (glossum /
# pyutilz / mlframe -- 8 raw hits, all FP, 0 after restriction):
#
# 1. Only STATEMENT-LEVEL (discarded) calls are flagged. A call whose result
#    is assigned/collected is routinely a "build coroutines now, gather
#    later" pattern (``tasks = [f(x) for x in xs]; await asyncio.gather(*tasks)``).
# 2. Only plain-name calls to an ``async def`` defined in the SAME module --
#    no cross-module type inference, no attribute calls.
# 3. The name must not be locally rebound in the enclosing function (a
#    function-local ``from x import count_tokens`` importing a SYNC function
#    legitimately shadows a same-named async method).


def scan_missed_await(root: Path,
                      exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                      ) -> list[Finding]:
    """Find statement-level calls to a same-module ``async def`` that are
    neither awaited nor wrapped in a task -- the coroutine is created and
    silently discarded, so the async function's body never executes.

    Severity: P1 (the operation silently does not happen; Python emits
    only a RuntimeWarning that most runs never surface).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        async_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)}
        if not async_names:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            rebound: set[str] = set()
            for n in ast.walk(func):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for alias in n.names:
                        rebound.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    rebound.add(n.id)
                elif isinstance(n, ast.arg):
                    rebound.add(n.arg)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not func:
                    # A nested def with the same name as a module-level async function is a
                    # legitimate local shadow (same idea as the local-import shadow above).
                    rebound.add(n.name)
            for stmt in ast.walk(func):
                if not isinstance(stmt, ast.Expr):
                    continue
                call = stmt.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                    continue
                name = call.func.id
                if name in async_names and name not in rebound:
                    findings.append(Finding(
                        check="missed_await",
                        severity="P1",
                        file=rel,
                        line=call.lineno,
                        snippet=_line_text(src_lines, call.lineno),
                        detail=(
                            f"`{name}(...)` is an async def in this module, called as a "
                            f"bare statement with the coroutine discarded -- its body "
                            f"never runs. Add `await` (or wrap in a task) so the "
                            f"operation actually executes."
                        ),
                    ))
    return findings


# --- sync-blocking call inside async def ----------------------------------
#
# Symmetric "sibling" of the missed-await scan above: that one catches "forgot to await a
# coroutine"; this one catches "used a synchronous blocking call inside a coroutine with no
# await/to_thread wrapper", which stalls the WHOLE event loop (every other in-flight coroutine),
# not just the current task. Confirmed real bug in the 2026-07-21 audit:
# openrouter_provider/_catalogue.py's `httpx.get()` (no await) was reachable from `async def
# generate()`/`generate_stream()` through a same-package helper chain, blocking the entire event
# loop for up to ~10s on the first call per process.

_BLOCKING_ATTR_CALLS = frozenset({"get", "post", "put", "delete", "patch", "request", "urlopen", "sleep", "run", "call", "check_call", "check_output"})
_BLOCKING_ROOT_HINTS = frozenset({"requests", "grequests", "urllib", "time", "subprocess"})
# httpx is intentionally excluded from _BLOCKING_ROOT_HINTS: httpx.AsyncClient's own .get()/.post()
# ARE legitimately awaited, so a bare root-name check would false-positive on the common case.


def _is_blocking_call(node: ast.Call) -> bool:
    """True if ``node`` looks like a synchronous blocking call (``requests.get(...)``,
    ``time.sleep(...)``, ``subprocess.run(...)``, bare ``httpx.get(...)`` -- NOT
    ``httpx.AsyncClient().get(...)``, which is a legitimate async call site normally awaited)."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in _BLOCKING_ATTR_CALLS:
        return False
    base = node.func.value
    while isinstance(base, ast.Attribute):
        base = base.value
    if isinstance(base, ast.Name):
        if base.id in _BLOCKING_ROOT_HINTS:
            return True
        if base.id == "httpx" and node.func.attr in ("get", "post", "put", "delete", "patch", "request"):
            return True
    return False


class _BlockingCallFinder(ast.NodeVisitor):
    """Collect blocking ``ast.Call`` nodes reachable in a function body that are NOT inside an
    ``await`` expression, tracking Await-ancestor state via an explicit depth counter rather than
    ``ast.walk`` (which has no ancestor information)."""

    def __init__(self) -> None:
        self.hits: list[ast.Call] = []
        self._await_depth = 0

    def visit_Await(self, node: ast.Await) -> None:
        """Track entry/exit of an ``await`` expression's subtree so calls inside it are not flagged."""
        self._await_depth += 1
        self.generic_visit(node)
        self._await_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        """Record ``node`` as a hit if it's a blocking call not currently nested under an ``await``."""
        if self._await_depth == 0 and _is_blocking_call(node):
            self.hits.append(node)
        self.generic_visit(node)

    # Don't descend into nested function/lambda scopes -- their own blocking calls are that
    # scope's own concern (a nested `def` might itself be sync, called via run_in_executor).
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Stop traversal at a nested sync function's own scope."""
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Stop traversal at a nested async function's own scope."""
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Stop traversal at a nested lambda's own scope."""
        return None


def scan_sync_blocking_in_async(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find synchronous blocking calls (``requests.*``, bare ``httpx.get/post/...``,
    ``time.sleep``, ``subprocess.run/call/check_call/check_output``) inside an ``async def`` body
    with no ``await``/``asyncio.to_thread`` wrapper -- this stalls the ENTIRE event loop (every
    other in-flight coroutine), not just the current task.

    Only checks calls made directly in the async function's own body (does not follow calls into
    helper functions -- same-file call-graph inlining is a known blind spot, kept out to avoid
    false positives from cross-file/library calls this scanner can't resolve).

    Severity: P1 (a real, confirmed-in-the-wild event-loop-stall bug class).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, ast.AsyncFunctionDef):
                continue
            finder = _BlockingCallFinder()
            for stmt in func.body:
                finder.visit(stmt)
            for call in finder.hits:
                assert isinstance(call.func, ast.Attribute)  # nosec B101 - _is_blocking_call only ever appends Calls whose .func is an Attribute; narrows the type for the access below, not a security check
                attr = call.func.attr
                findings.append(Finding(
                    check="sync_blocking_in_async",
                    severity="P1",
                    file=rel,
                    line=call.lineno,
                    snippet=_line_text(src_lines, call.lineno),
                    detail=(
                        f"`...{attr}(...)` is a synchronous blocking call inside `async def "
                        f"{func.name}` with no `await`/`asyncio.to_thread` wrapper -- stalls the "
                        "WHOLE event loop (every other in-flight coroutine), not just this task."
                    ),
                ))
    return findings
