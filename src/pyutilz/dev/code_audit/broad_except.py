"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- broad-except swallow (wave 16) -------------------------------------


_LOG_METHODS = frozenset({"warning", "error", "critical", "exception", "warn"})


def _has_log_call(stmts: list[ast.stmt]) -> bool:
    """True if ``stmts`` contains a logger.warning/error/critical/exception call."""
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in _LOG_METHODS:
                    return True
    return False


def _has_raise(stmts: list[ast.stmt]) -> bool:
    """True if ``stmts`` contains a ``raise`` statement (including a bare re-raise)."""
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Raise):
                return True
    return False


def _has_debug_only_log_call(stmts: list[ast.stmt]) -> bool:
    """True if ``stmts`` contains a ``logger.debug(...)`` call and no
    warning/error/critical/exception/warn call. Debug is off by default in
    production, so operators still get no signal from it -- but it's a
    materially different (weaker, downgradeable) signal than TRUE silence
    (``except: pass``), since it's visible the moment someone turns debug
    logging on for diagnosis."""
    if _has_log_call(stmts):
        return False
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "debug":
                return True
    return False


def _is_silent_swallow(handler: ast.ExceptHandler) -> bool:
    """True if the handler body is silent: only pass/continue/return/break
    AND has no log call (of any level, including debug) AND has no re-raise."""
    if _has_log_call(handler.body):
        return False
    if _has_debug_only_log_call(handler.body):
        return False
    if _has_raise(handler.body):
        return False
    # If body contains a meaningful side-effect (assign, function call other
    # than logger), it's NOT a silent swallow.
    for s in handler.body:
        if isinstance(s, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(s, ast.Return):
            continue
        return False
    return True


def _is_broad_except(handler: ast.ExceptHandler) -> bool:
    """True if the except clause catches Exception or is bare."""
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
        return True
    if isinstance(handler.type, ast.Tuple):
        return any(isinstance(el, ast.Name) and el.id in {"Exception", "BaseException"} for el in handler.type.elts)
    return False


# Best-effort filesystem / process operations whose failure is legitimately
# swallowed in production (the file may not exist, the dir may already be
# cleaned up, the process may have already exited). Suppress the broad-except
# warning when the try body is a single call to one of these.
_BEST_EFFORT_OPS = frozenset({
    # filesystem
    "chmod", "unlink", "remove", "rmdir", "makedirs", "mkdir", "rmtree",
    "close", "flush",
    # process
    "kill", "terminate", "wait",
    # logging / metrics teardown
    "stop", "shutdown", "join",
})


def _try_body_is_imports_only(try_body: list[ast.stmt]) -> bool:
    """True if every statement in the try body is an Import / ImportFrom
    (optional-dep guard pattern). These broad-except blocks are legitimate
    -- the swallow is the entire point of the import guard."""
    if not try_body:
        return False
    for s in try_body:
        if not isinstance(s, (ast.Import, ast.ImportFrom)):
            return False
    return True


def _try_body_is_best_effort_op(try_body: list[ast.stmt]) -> bool:
    """True if the try body is a single call (or attribute assignment) to a
    documented best-effort op (``os.chmod``, ``proc.kill``, ``file.close``,
    etc.). These swallows are intentional and don't degrade observability."""
    if len(try_body) != 1:
        return False
    s = try_body[0]
    # ``os.chmod(path, mode)`` etc.
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
        call = s.value
        if isinstance(call.func, ast.Attribute) and call.func.attr in _BEST_EFFORT_OPS:
            return True
        if isinstance(call.func, ast.Name) and call.func.id in _BEST_EFFORT_OPS:
            return True
    return False


_DOCUMENTED_RATIONALE_MARKERS = ("nosec", "opportunistic", "best-effort", "best effort")


def _handler_has_documented_rationale(handler: ast.ExceptHandler, src_lines: list[str]) -> bool:
    """True if the except line itself, or the handler body's own comment/docstring lines, name a
    documented reason the swallow is intentional (a ``# nosec ...`` bandit-suppression comment, or
    the words "opportunistic"/"best-effort" -- this project's own established vocabulary for "this
    path is optional, any failure here is expected and handled by falling through"). A human
    already made and recorded this call; re-flagging it every scan is pure noise, not a new signal.
    Matches only the handler's own line + body span (not the whole enclosing function) so an
    unrelated nosec/opportunistic comment elsewhere in a large function doesn't accidentally exempt
    a genuinely-undocumented swallow a few lines below it."""
    start = handler.lineno
    end = max((getattr(s, "end_lineno", s.lineno) or s.lineno) for s in handler.body) if handler.body else start
    window = "\n".join(_line_text(src_lines, i) for i in range(start, end + 1)).lower()
    return any(marker in window for marker in _DOCUMENTED_RATIONALE_MARKERS)


def scan_broad_except_swallows(root: Path,
                               exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                               ) -> list[Finding]:
    """Find broad ``except Exception:`` / ``except:`` handlers whose body
    silently passes/continues/returns without any logger.warning/error
    call or re-raise. These are the classic silent-data-drop sites.

    Severity P1 by default; downgrade to Low for import-guard shapes
    detected via the ``ImportError`` / ``ModuleNotFoundError`` exception
    type (those should already short-circuit via the broad-except check,
    but defensive narrowing helps).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Suppress whole Try if the body is one of the documented-safe
            # shapes (import guard / single best-effort op). The handler
            # body's silent swallow is intentional in these cases.
            if _try_body_is_imports_only(node.body):
                continue
            if _try_body_is_best_effort_op(node.body):
                continue
            for handler in node.handlers:
                if not _is_broad_except(handler):
                    continue
                if not _is_silent_swallow(handler):
                    continue
                if _handler_has_documented_rationale(handler, src_lines):
                    continue
                kind = "bare except" if handler.type is None else "except Exception"
                findings.append(Finding(
                    check="broad_except_swallow",
                    severity="P1",
                    file=rel,
                    line=handler.lineno,
                    snippet=_line_text(src_lines, handler.lineno),
                    detail=(
                        f"{kind} with silent body (pass/continue/return, "
                        f"no logger.warning/error, no re-raise). "
                        f"Operators get no signal when this fires."
                    ),
                ))
    return findings
