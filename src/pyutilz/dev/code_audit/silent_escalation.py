"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text
from .broad_except import _has_log_call, _has_raise, _is_broad_except

# --- logged-but-not-escalated except ---------------------------------------

# Common names for a caller-visible collection an except handler is expected
# to append a failure marker to, when the surrounding code follows a
# "collect problems, report them at the end" convention (as opposed to
# failing fast). Projects with their own naming convention pass their own
# set via the ``escalation_attrs`` kwarg.
DEFAULT_ESCALATION_ATTRS: frozenset[str] = frozenset({
    "validation_errors", "errors", "warnings", "issues", "problems", "failures",
})


def _escalates_to(handler: ast.ExceptHandler, escalation_attrs: frozenset[str]) -> bool:
    """True if the handler body appends/extends one of ``escalation_attrs``
    (matched as the attribute immediately before ``.append``/``.extend``,
    e.g. ``result.validation_errors.append(...)`` or
    ``self.errors.extend(...)``)."""
    for node in ast.walk(handler):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend")):
            continue
        base = node.func.value
        if isinstance(base, ast.Attribute) and base.attr in escalation_attrs:
            return True
        if isinstance(base, ast.Name) and base.id in escalation_attrs:
            return True
    return False


def scan_log_only_except(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    escalation_attrs: frozenset[str] = DEFAULT_ESCALATION_ATTRS,
) -> list[Finding]:
    """Find broad ``except`` handlers that log the failure but never
    escalate it to a caller-visible collection and never re-raise.

    Complements ``scan_broad_except_swallows``: that scanner flags
    handlers with NO log call at all. This one specifically targets the
    handler that DOES log (so it reads as "handled") but the log line is
    the only trace of the failure -- a caller inspecting the return
    value (or a persisted "was this successful" flag/counter derived
    from that value) has no way to learn anything failed. Concretely:
    a batch-save helper that catches a per-item DB error, logs a
    warning, and continues -- when the caller's success flag is computed
    from an in-memory list that was never told about the failure, the
    caller (and anything it persists) reports success at a lower true
    count.

    Only applies within files that reference at least one of
    ``escalation_attrs`` (a caller-visible list/collection naming
    convention, e.g. ``validation_errors``) SOMEWHERE -- if the
    convention isn't used at all in this file, it isn't the "collect
    problems, report at the end" pattern and silence is a design
    choice, not a gap. Scoped per-file (not per-function) to keep the
    walk a single flat pass over ``ast.Try`` nodes, matching the other
    scanners in this package -- a file with one function using the
    convention and an unrelated function that doesn't can produce an
    occasional false positive in the unrelated function; this scanner
    is advisory (P2) and meant to be reviewed, not auto-failed on.

    Severity: P2 (an operator sees a log line; a caller reading only the
    return value / persisted flag does not).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace")
        if not any(attr in src_lines for attr in escalation_attrs):
            continue
        lines = src_lines.splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if not _is_broad_except(handler):
                    continue
                if not _has_log_call(handler.body):
                    continue  # scan_broad_except_swallows already covers the no-log case
                if _has_raise(handler.body):
                    continue
                if _escalates_to(handler, escalation_attrs):
                    continue
                findings.append(Finding(
                    check="log_only_except",
                    severity="P2",
                    file=rel,
                    line=handler.lineno,
                    snippet=_line_text(lines, handler.lineno),
                    detail=(
                        "except handler logs the failure but never appends to "
                        f"a {sorted(escalation_attrs)} collection or re-raises -- "
                        "a caller reading only the return value / a persisted "
                        "success flag has no way to learn this failed, even "
                        "though this file otherwise uses that "
                        "collect-and-report convention."
                    ),
                ))
    return findings
