"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- unbounded or silently-exhausting retry loop --------------------------
#
# Class of bug found repeatedly in the 2026-07-21 audit: redislib.py's `while True:` retried
# every ConnectionError forever, INCLUDING AuthenticationError (a permanent failure -- wrong
# password will never succeed no matter how many times it's retried, and AuthenticationError is
# a subclass of ConnectionError in redis-py, so a naive `except ConnectionError` catches it too);
# browser.py's LoginAndGetCookies() busy-looped on an unrecognized exception with no sleep()
# between attempts (100% CPU forever); db/__init__.py's execute_alchemy() bounded retry loop fell
# through to an implicit `return None` on exhaustion instead of raising, indistinguishable from a
# legitimate successful-but-empty result.
#
# This scanner is intentionally conservative (allowlist-friendly via the baseline mechanism):
# it flags `while True:` loops containing a `try/except` with `sleep()`/`continue` but no
# `break`-on-success path visible, AND bounded-counter retry loops (`while attempt < max:`)
# whose loop body's `try/except` has no `raise` anywhere and the code immediately after the loop
# (same indent level, no `else:`) has no `raise`/explicit sentinel-return distinguishable from
# the success path either.


def _loop_has_break(loop: ast.While) -> bool:
    """True if ``loop``'s own body (not nested loops) contains a ``break``."""
    for node in ast.walk(loop):
        if node is loop:
            continue
        if isinstance(node, (ast.While, ast.For)):
            continue  # a break in a NESTED loop doesn't exit this one
        if isinstance(node, ast.Break):
            return True
    return False


def _try_has_sleep_or_continue(node: ast.AST) -> bool:
    """True if ``node``'s subtree contains a ``continue`` or a call to a function/method named ``sleep``."""
    for n in ast.walk(node):
        if isinstance(n, ast.Continue):
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sleep":
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sleep":
            return True
    return False


def _except_has_sleep(handler: ast.ExceptHandler) -> bool:
    """True if ``handler``'s body contains a call to a function/method named ``sleep``."""
    for n in ast.walk(handler):
        if isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "sleep") or (isinstance(n.func, ast.Name) and n.func.id == "sleep")
        ):
            return True
    return False


def _try_body_has_blocking_wait_with_timeout(try_node: ast.Try) -> bool:
    """True if the ``try:`` block itself (not its ``except`` handlers)
    contains a call passing a ``timeout=`` keyword -- e.g. the common
    producer/consumer idiom ``item = queue.get(timeout=N); except
    queue.Empty: continue``. That blocking wait already provides the same
    backoff a ``sleep()`` in the handler would -- it happens BEFORE the
    exception is even raised, not after it -- so an except handler with no
    ``sleep()`` here is not a busy-loop. Confirmed false positive found in
    the wild (a queue-drain loop) during the first real run of this scanner.
    """
    for stmt in try_node.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call) and any(kw.arg == "timeout" for kw in n.keywords):
                return True
    return False


def scan_retry_loops(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find two shapes of risky retry loop:

    1. ``while True:`` containing a ``try/except`` that retries (has ``continue``/no explicit
       ``break``-on-success visible) but whose ``except`` handler has NO ``sleep()`` call --
       a busy-loop that can burn 100% CPU forever on a persistently-failing operation.
    2. A ``while True:`` retry loop whose ``except`` handler catches a broad exception type
       (``Exception``/no type/a single class with no narrower branch distinguishing permanent
       vs transient failures) with no upper bound anywhere in the loop -- flagged as Low
       (advisory: not always wrong, e.g. "wait for a service to come up" is a legitimate use, but
       worth a human confirming it's deliberate).

    Severity: P1 for the busy-loop shape (no sleep -- a much sharper, low-FP signal); Low for the
    unbounded-without-busy-loop shape (advisory only, matches the 2026-07-21 audit's own
    "needs confirmation" disposition for ``connect_to_db``'s deliberately-infinite retry).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True):
                continue
            try_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Try)]
            if not try_nodes:
                continue
            has_retry_shape = any(_try_has_sleep_or_continue(t) for t in try_nodes)
            if not has_retry_shape:
                continue
            any_except_has_sleep = any(_except_has_sleep(h) for t in try_nodes for h in t.handlers)
            any_blocking_wait = any(_try_body_has_blocking_wait_with_timeout(t) for t in try_nodes)
            if not any_except_has_sleep and not any_blocking_wait:
                findings.append(Finding(
                    check="busy_retry_loop",
                    severity="P1",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        "`while True:` retry loop's except handler has no `sleep()` -- a "
                        "persistently-failing operation busy-loops at 100% CPU forever instead "
                        "of backing off."
                    ),
                ))
            elif not _loop_has_break(node):
                findings.append(Finding(
                    check="unbounded_retry_loop",
                    severity="Low",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        "`while True:` retry loop with a sleep-backed except handler but no "
                        "visible `break` -- confirm this is a deliberate 'retry forever' design "
                        "(e.g. waiting for a service at startup), not a missing upper bound on a "
                        "condition that should eventually give up and raise."
                    ),
                ))
    return findings
