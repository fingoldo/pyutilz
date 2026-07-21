"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: ``logger.exception()``
must only ever be called from WITHIN an active ``except`` handler (lexically) -- calling it
anywhere else means ``sys.exc_info()`` is empty or stale by the time it runs, silently logging
"NoneType: None" instead of the real traceback.

Class this catches: this generalizes the ``system/monitoring.py`` bug fixed earlier this round,
where a heartbeat callback captured/logged an exception from a context that could run on a
DIFFERENT thread (or after the original handler had already returned) than the one that caught
it -- ``sys.exc_info()`` is thread-local and stack-scoped, so ``logger.exception()`` called from
that deferred context produced a useless, traceback-less log line while looking, at a glance,
like it was doing the right thing.

Note: an earlier, broader design for this test also flagged ``except`` handlers that log the
bound exception variable via ``logger.warning``/``error``/``critical`` WITHOUT a paired
``logger.exception()``/``exc_info=``. Implementing and running that check against this codebase
surfaced 38 matches; spot-checking a representative sample (``database.db``'s ``DuplicateTable``
vs ``InternalError`` handlers logging the identical way except for severity+traceback,
``web.web``'s deliberately-terse retry-loop warning, ``cloud.cloud``'s S3-download retry) showed
the overwhelming majority are a DELIBERATE, existing convention distinguishing expected/routine
failures (log tersely, no traceback needed) from unexpected/serious ones (already correctly using
``.exception()``) -- not a bug class an AST shape check can reliably tell apart from intentional
style. That check was dropped rather than shipped as a high-noise gate; this file keeps only the
half that measured near-zero false positives against the current codebase.

Mechanism: AST-walk every ``.py`` file for ``ast.Call`` nodes shaped ``<logger>.exception(...)``
where ``<logger>`` is a bare name or attribute access ending in ``logger``/``log``/``_logger``
(filters out unrelated ``.exception()`` methods on other objects, e.g.
``concurrent.futures.Future.exception()`` / tenacity's ``RetryCallState.outcome.exception()``,
which produced a false positive during design). For each match, walk up the PARENT chain (built
via a preliminary ``ast.iter_child_nodes`` pass) until hitting either an ``ast.ExceptHandler``
(pass) or a ``FunctionDef``/``AsyncFunctionDef``/``Module`` boundary with no ``ExceptHandler`` in
between (fail) -- a lexically nested function called asynchronously (a thread-pool submission, a
callback, a scheduled task) is exactly the shape that produces a stale/empty ``sys.exc_info()``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent

_LOGGER_NAMES = {"logger", "log", "_logger"}


def _is_logger_receiver(value: ast.expr) -> bool:
    if isinstance(value, ast.Name):
        return value.id.lower() in _LOGGER_NAMES
    if isinstance(value, ast.Attribute):
        return value.attr.lower() in _LOGGER_NAMES
    return False


def _set_parents(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _has_enclosing_handler(node: ast.AST) -> bool:
    cur: object = node
    while cur is not None:
        if isinstance(cur, ast.ExceptHandler):
            return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return False
        cur = getattr(cur, "parent", None)
    return False


def _find_exception_calls_outside_handlers(root: Path) -> list[str]:
    out: list[str] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        _set_parents(tree)
        rel = py.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exception"):
                continue
            if not _is_logger_receiver(node.func.value):
                continue
            if not _has_enclosing_handler(node):
                out.append(f"{rel}:{node.lineno}")
    return out


def test_no_logger_exception_calls_outside_an_active_handler():
    found = _find_exception_calls_outside_handlers(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} logger.exception() call(s) with no enclosing except handler in the "
            f"same lexical function. sys.exc_info() is thread-local and stack-scoped -- calling "
            f"logger.exception() from a nested function that gets invoked asynchronously (a "
            f"thread-pool submission, a deferred callback, a different thread) after the "
            f"original handler's frame is gone produces a useless 'NoneType: None' traceback "
            f"instead of the real one. Either call logger.exception() synchronously inside the "
            f"except block, or capture the exception object explicitly (`except Exception as e:`) "
            f"and pass it to the deferred callback so it can log with `exc_info=e`:\n  " + "\n  ".join(found)
        )
