"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

_LOG_METHODS = ("warning", "error", "exception", "critical")
_THROTTLE_HINTS = ("throttle", "rate_limit", "ratelimit", "debounce")


def _call_name(node: ast.AST) -> str | None:
    """Return a Call's callable name (bare ``Name.id`` or ``Attribute.attr``), else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _guard_looks_throttled(test: ast.expr) -> bool:
    """True if an ``if`` test guarding a log call plausibly already throttles it: calls a
    helper whose name contains a throttle-shaped hint, or uses a modulo/interval-counter idiom
    (``if i % N == 0``, a common cheap throttle)."""
    for node in ast.walk(test):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name and any(h in name.lower() for h in _THROTTLE_HINTS):
                return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return True
    return False


def _is_log_call(node: ast.AST) -> str | None:
    """Return the log method name ('warning'/'error'/...) if this Call node is
    ``<something ending in 'log'/'logger'>.<method>(...)``, else None."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr not in _LOG_METHODS:
        return None
    receiver = node.func.value
    receiver_name = None
    if isinstance(receiver, ast.Name):
        receiver_name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        receiver_name = receiver.attr
    if receiver_name is None:
        return None
    low = receiver_name.lower()
    if low.endswith("log") or low.endswith("logger") or low == "_log":
        return node.func.attr
    return None


def _visit_if_aware(
    node: ast.AST,
    loop_depth: int,
    guarded: bool,
    findings: list[Finding],
    rel: str,
    src_lines: list[str],
) -> None:
    """Manual recursive descent (not ast.walk) so If nodes can pass the "throttle-guarded" flag
    to their `body` only (not `orelse`), and loop nodes bump depth for their body/orelse but not
    their own target/iter/test expressions. Takes findings/rel/src_lines as explicit params
    (not closure captures) so this can be a plain module-level function reused across files."""
    method = _is_log_call(node)
    if method is not None and loop_depth > 0 and not guarded:
        assert isinstance(node, ast.Call)  # guaranteed by _is_log_call returning non-None
        findings.append(Finding(
            check="unthrottled_hot_loop_log",
            severity="P2",
            file=rel,
            line=node.lineno,
            snippet=_line_text(src_lines, node.lineno),
            detail=(
                f"log.{method}(...) inside a loop (depth {loop_depth}) with no apparent throttle guard "
                f"-- every iteration that hits this line logs, compounding into spam under load. "
                f"Guard with a throttle helper (e.g. this project's own rate-limit debounce pattern) "
                f"or demote to log.debug()."
            ),
        ))
    if isinstance(node, (ast.For, ast.AsyncFor)):
        for child in (node.target, node.iter, *node.body, *node.orelse):
            if child is not None:
                _visit_if_aware(child, loop_depth + 1, False, findings, rel, src_lines)
    elif isinstance(node, ast.While):
        for child in (node.test, *node.body, *node.orelse):
            _visit_if_aware(child, loop_depth + 1, False, findings, rel, src_lines)
    elif isinstance(node, ast.If):
        child_guarded = guarded or _guard_looks_throttled(node.test)
        for child in node.body:
            _visit_if_aware(child, loop_depth, child_guarded, findings, rel, src_lines)
        for child in node.orelse:
            _visit_if_aware(child, loop_depth, guarded, findings, rel, src_lines)
        _visit_if_aware(node.test, loop_depth, guarded, findings, rel, src_lines)
    else:
        for child in ast.iter_child_nodes(node):
            _visit_if_aware(child, loop_depth, guarded, findings, rel, src_lines)


def scan_unthrottled_hot_loop_log(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a ``log.warning``/``log.error`` (etc.) call inside a ``for``/``while`` loop with no
    apparent rate-limiting guard -- a per-item log line that compounds into log-flood spam the
    moment every item in a large batch hits the same condition (e.g. a systemic upstream outage).

    Each individual call looks reasonable in isolation ("log the error when it happens"); the
    compounding-under-load behavior only shows up during an actual incident, which code review
    rarely simulates. Real example: ``job_details_shared.py``'s two ``ErrorCat.OTHER`` warning
    sites fired unconditionally per-item inside a batch loop, right next to an established
    ``_log_throttle`` debounce helper already used elsewhere in the same codebase for the
    structurally identical ``scan_loop`` case (2026-07-21 full-audit, observability-Low, closed
    in the gap-pass).

    A log call is considered guarded (not flagged) if its nearest enclosing ``if`` test either
    calls a function whose name contains a throttle-shaped hint (``throttle``, ``rate_limit``,
    ``debounce``, case-insensitive -- matches this project's own ``_log_throttle`` and similarly
    named helpers elsewhere without per-project configuration) or uses a modulo expression
    (``i % N``), a common cheap "every Nth iteration" throttle idiom.

    Severity: P2 (usually a hygiene/observability issue, not a correctness bug -- but can degrade
    into real operational pain: log-volume-driven disk fill, alerting fatigue, or a downstream
    log-aggregation cost spike during exactly the outage an operator most needs signal, not noise).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        _visit_if_aware(tree, 0, False, findings, rel, src_lines)
    return findings
