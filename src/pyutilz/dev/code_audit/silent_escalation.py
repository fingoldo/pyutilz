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


_ESCALATION_METHOD_NAMES = frozenset({"append", "extend", "warn", "add_error", "add_warning"})


def _target_name(node: ast.AST) -> str | None:
    """The identifier a target expression is ultimately named by: the
    attribute name for ``x.error``, the bound name for a plain variable, or
    the string key for ``d["errors"]``. None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _looks_like_error_target(node: ast.AST) -> bool:
    """True when a target's name itself signals an error-collection
    convention this project didn't declare via ``escalation_attrs`` -- e.g.
    ``stats["errors"] += 1``, ``total_errors += len(batch)``,
    ``result.error = str(e)``. A substring match on "error" is deliberately
    broad: false negatives here (a genuinely unescalated failure) are worse
    than the rare false positive (an unrelated ``error`` in a variable
    name that isn't actually this handler's escalation path)."""
    name = _target_name(node)
    return name is not None and "error" in name.lower()


def _escalates_to(handler: ast.ExceptHandler, escalation_attrs: frozenset[str]) -> bool:
    """True if the handler body makes the failure observable to a caller
    through any of the escalation conventions seen in practice:

    - ``.append()``/``.extend()`` onto ``escalation_attrs`` or an
      error-named collection (``result.validation_errors.append(...)``,
      ``stats["errors"].append(...)``).
    - ``.warn()``/``.add_error()``/``.add_warning()`` on any object --
      distinct naming convention, method name alone is the signal.
    - An augmented increment of an error-named counter
      (``stats["errors"] += 1``, ``total_errors += len(batch)``).
    - A plain assignment to an error-named variable/attribute/key
      (``result.error = str(e)``, ``error_message = str(e)``) -- covers
      the common "stash it in a local, persist it after the loop" shape.
    - A ``return`` of a falsy sentinel (``False``, a tuple containing
      ``False``) or a dict literal with an error-named key
      (``return {"error": str(e)}``) -- the Phase0-style bool/dict-return
      escalation contract.
    """
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _ESCALATION_METHOD_NAMES:
            if node.func.attr in ("warn", "add_error", "add_warning"):
                return True
            base = node.func.value
            if isinstance(base, ast.Attribute) and base.attr in escalation_attrs:
                return True
            if isinstance(base, ast.Name) and base.id in escalation_attrs:
                return True
            if _looks_like_error_target(base):
                return True
        elif isinstance(node, ast.AugAssign) and _looks_like_error_target(node.target):
            return True
        elif isinstance(node, ast.Assign) and any(_looks_like_error_target(t) for t in node.targets):
            return True
        elif isinstance(node, ast.Return) and node.value is not None:
            v = node.value
            if isinstance(v, ast.Constant) and v.value is False:
                return True
            if isinstance(v, ast.Tuple) and any(isinstance(e, ast.Constant) and e.value is False for e in v.elts):
                return True
            if isinstance(v, ast.Dict) and any(isinstance(k, ast.Constant) and isinstance(k.value, str) and "error" in k.value.lower() for k in v.keys):
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
