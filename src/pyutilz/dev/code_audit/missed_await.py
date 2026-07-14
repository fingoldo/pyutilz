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
