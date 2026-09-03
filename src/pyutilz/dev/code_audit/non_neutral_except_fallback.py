"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- non-neutral literal substituted by a silent except handler ----------
#
# Sixteen findings, one shape, in the 2026-09-01 mlframe audit. A handler catches an exception, returns or
# assigns a literal, and logs at DEBUG (which production logging does not emit) or not at all. The substituted
# value is then indistinguishable from a real result -- and it is routinely non-neutral in the direction that
# DISABLES the very check it feeds:
#
# * `_max_err = 0.0` on a failed max-error computation is the BEST possible max error, so `_max_err > 5*y_std`
#   became unconditionally False and switched a collapse sensor off -- in exactly the situations (shape
#   mismatch, object-dtype predictions) that produce the failure.
# * `return True` from a failed VRAM probe is the value that ALLOWS the upload, removing OOM protection
#   precisely when the device is too unhealthy to answer.
# * `return -np.inf` for a failed solve is the value that GUARANTEES rejection, so a driver fault silently
#   discarded a good result.
# * `0.0` for a failed correlation reads as "uncorrelated" -- a claim, not an admission of ignorance.
#
# The test to apply at the call site: if this value were wrong, would anything downstream notice? When the
# answer is no, the failure has to be audible.
#
# Distinct from `log_only_except` (which asks whether a handler ESCALATES what it logged) and from
# `broad_except_swallow` (which asks how broadly it catches): this one keys on the substituted VALUE, and fires
# on a NARROW handler too. `None` is exempt -- it is the one substitution that reads as "no answer" rather than
# as an answer. An `ImportError`-only handler is exempt: a genuinely absent optional dependency is a permanent,
# expected condition, and its substitution is the intended answer rather than a masked failure.

DEFAULT_AUDIBLE_LOG_METHODS: frozenset[str] = frozenset({"warning", "error", "exception", "critical", "warn"})
# `print` counts. The concern this check raises is that the substituted value is
# INDISTINGUISHABLE downstream from a real result, and a line on stdout distinguishes it -- for a
# CLI script that is the notification channel, not a lapse. Three handlers in a harvest script
# that print exactly what they substituted and why were reported as silent.
DEFAULT_AUDIBLE_FUNCTIONS: frozenset[str] = frozenset({"log_throttle", "print"})


# Exceptions that state a DETERMINATE FACT about the input rather than reporting that an operation which
# could have gone either way went wrong: this string is not a number, this key/index/attribute is not there,
# this file does not exist, these bytes are not valid UTF-8. When such an exception is the ONLY thing a
# handler catches and the guarded body is a single expression, the handler is the negative branch of a total
# function ("parse this, or say it is unparseable"), and its substituted value IS the answer -- there is no
# masked failure for a downstream consumer to be misled about. `OSError`, `subprocess.CalledProcessError`
# and bare `Exception` are deliberately absent: those are the environment-failure spellings this check
# exists for (py-ci-shared's `_loc` returning 0 LOC on OSError silently exempts the file from a size gate).
_DETERMINATE_NEGATIVE_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "LookupError",
        "AttributeError",
        "NameError",
        "ZeroDivisionError",
        "ArithmeticError",
        "OverflowError",
        "FloatingPointError",
        "RecursionError",
        "StopIteration",
        "StopAsyncIteration",
        "UnicodeError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeTranslateError",
        "FileNotFoundError",
        "JSONDecodeError",
        "SyntaxError",
    }
)


def _handler_exception_names(handler: ast.ExceptHandler) -> set[str]:
    """The bare names of the exception classes a handler catches (``orjson.JSONDecodeError`` -> ``JSONDecodeError``)."""
    if handler.type is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(handler.type):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    # `except orjson.JSONDecodeError` walks to both the Attribute and the `orjson` Name; the module
    # component is not an exception class, so drop any name that is only a prefix of a dotted one.
    for node in ast.walk(handler.type):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.discard(node.value.id)
    return names


def _is_import_error_only(handler: ast.ExceptHandler) -> bool:
    """True when the handler catches ImportError / ModuleNotFoundError and nothing broader."""
    names = _handler_exception_names(handler)
    return bool(names) and names <= {"ImportError", "ModuleNotFoundError"}


def _is_determinate_negative_only(handler: ast.ExceptHandler) -> bool:
    """True when every exception the handler catches states a determinate fact about the input."""
    names = _handler_exception_names(handler)
    return bool(names) and names <= _DETERMINATE_NEGATIVE_EXCEPTIONS


#: How many straight-line statements a total-function guard may hold. The measured shapes are one
#: (``try: return float(s)``) or two (``try: float(s)`` then ``return True`` -- the ``is_float``
#: predicate spelling). Three leaves a little room without admitting a multi-step operation.
_MAX_TOTAL_FUNCTION_GUARD_STMTS = 3


def _is_single_expression_guard(try_body: list) -> bool:
    """True when the guarded body is a short run of statements that only evaluate/bind expressions.

    Paired with ``_is_determinate_negative_only``: together they identify a total-function guard
    (``try: return float(s)`` / ``except ValueError: return 0.0``, or its predicate spelling
    ``try: float(s); return True`` / ``except ValueError: return False``) rather than a multi-step
    operation whose partial failure the handler is papering over. Any control flow in the body --
    a loop, an ``if``, a nested ``try`` -- disqualifies it: then the handler can fire from several
    different places and the substituted value no longer means one determinate thing.
    """
    if not try_body or len(try_body) > _MAX_TOTAL_FUNCTION_GUARD_STMTS:
        return False
    return all(isinstance(stmt, (ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)) for stmt in try_body)


def _guards_an_import(try_body: list) -> bool:
    """True when the guarded body imports something.

    An optional dependency that is absent, or present but unusable on this host (numba refusing to
    load against the installed llvmlite, cupy raising on a driverless box, a lazy import broken by a
    circular-import bootstrap), is a PERMANENT, EXPECTED condition, and ``_HAS_NUMBA = False`` /
    ``return False`` is the intended answer to "is it available", not a masked measurement. The
    existing ``_is_import_error_only`` exemption covers only the narrow spelling; in practice these
    probes catch broadly ON PURPOSE, precisely because a broken-but-installed dependency raises
    something other than ImportError -- which is why the broad spelling dominated the measured
    false positives (12 of 30 sampled mlframe sites, 9 of 41 pyutilz sites).
    """
    for stmt in try_body:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return True
    return False


def _is_audible(handler: ast.ExceptHandler, log_methods: frozenset[str], functions: frozenset[str]) -> bool:
    """True when the handler re-raises, or logs at warning level or above."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in log_methods:
            return True
        if isinstance(fn, ast.Name) and fn.id in functions:
            return True
    return False


def _describe_literal(value: ast.AST) -> str | None:
    """Name the substituted literal when it is one worth flagging, else None."""
    if isinstance(value, ast.Constant):
        # `None` reads as "no answer" rather than as a measurement, so it is not a finding.
        if value.value is None:
            return None
        return repr(value.value) if isinstance(value.value, (bool, int, float, str)) else None
    # `-np.inf` / `np.inf` -- the classic "guarantees the comparison's outcome" substitution.
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        inner = value.operand
        if isinstance(inner, ast.Attribute) and inner.attr in {"inf", "Inf", "infty"}:
            return "-inf"
    if isinstance(value, ast.Attribute) and value.attr in {"inf", "Inf", "infty"}:
        return "inf"
    return None


def _own_nodes(handler: ast.ExceptHandler) -> "Iterator[ast.AST]":
    """Walk the handler body WITHOUT descending into nested function/lambda scopes.

    A `def cb(): return 0.0` defined inside the handler is a callback the handler registers, not
    a value the handler substitutes for the failed computation.
    """
    todo: list[ast.AST] = list(handler.body)
    while todo:
        node = todo.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        todo.extend(ast.iter_child_nodes(node))


def _target_name(target: ast.AST) -> str:
    """Best-effort display name for an assignment target."""
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return "<target>"


def _substituted_literal(handler: ast.ExceptHandler) -> str | None:
    """A short description of the non-neutral literal this handler substitutes, or None.

    Candidates are ordered by source position so the reported handler line and the quoted value
    always describe the FIRST substitution -- ``ast.walk`` is breadth-first and would pick an
    arbitrary one when a handler has several.
    """
    candidates: list[tuple[tuple[int, int], str]] = []
    for node in _own_nodes(handler):
        pos = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        if isinstance(node, ast.Return) and node.value is not None:
            desc = _describe_literal(node.value)
            if desc is not None:
                candidates.append((pos, f"returns {desc}"))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            # `_max_err: float = 0.0` -- the annotated spelling this project's conventions mandate.
            desc = _describe_literal(node.value)
            if desc is not None:
                candidates.append((pos, f"assigns {_target_name(node.target)} = {desc}"))
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)) and len(target.elts) == len(node.value.elts):
                for elt_target, elt_value in zip(target.elts, node.value.elts):
                    desc = _describe_literal(elt_value)
                    if desc is not None:
                        candidates.append((pos, f"assigns {_target_name(elt_target)} = {desc}"))
                        break
                continue
            desc = _describe_literal(node.value)
            if desc is not None:
                candidates.append((pos, f"assigns {_target_name(target)} = {desc}"))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]


def scan_non_neutral_except_fallback(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    audible_log_methods: frozenset[str] = DEFAULT_AUDIBLE_LOG_METHODS,
    audible_functions: frozenset[str] = DEFAULT_AUDIBLE_FUNCTIONS,
) -> list[Finding]:
    """Flag an except handler that substitutes a non-neutral literal without saying so above debug level.

    ``audible_functions`` names bare-call throttled-logging helpers a project uses on hot paths (a plain warning
    there would spam), so a handler reaching one counts as audible.

    Severity: P1. The substituted value flows onward as if it were a measurement, and frequently it is the exact
    value that disables the check consuming it.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        # Walk Try nodes (not bare handlers) so each handler is judged together with the body it guards.
        # ``ast.TryStar`` (3.11+) is fetched by name so this stays importable on Python 3.8.
        _try_types: tuple = (ast.Try,) if getattr(ast, "TryStar", None) is None else (ast.Try, ast.TryStar)  # type: ignore[attr-defined]
        for try_node in [n for n in ast.walk(tree) if isinstance(n, _try_types)]:
            for handler in try_node.handlers:
                if _is_import_error_only(handler) or _is_audible(handler, audible_log_methods, audible_functions):
                    continue
                if _guards_an_import(try_node.body):
                    continue
                if _is_determinate_negative_only(handler) and _is_single_expression_guard(try_node.body):
                    continue
                desc = _substituted_literal(handler)
                if desc is None:
                    continue
                findings.append(
                    Finding(
                        check="non_neutral_except_fallback",
                        severity="P1",
                        file=rel,
                        line=handler.lineno,
                        snippet=_line_text(src_lines, handler.lineno),
                        detail=(
                            f"handler {desc} without logging above debug level, so the substituted value is "
                            f"indistinguishable downstream from a real result. Ask whether anything would notice if "
                            f"it were wrong -- and note that such a value is often non-neutral in the direction that "
                            f"DISABLES the check it feeds (0.0 for a max error, True for a permission guard, -inf "
                            f"for a score being minimised), which is how a safety mechanism switches itself off in "
                            f"exactly the conditions that trip it. Use a warning (throttled on a hot path), a "
                            f"NaN/None that reads as 'unknown' rather than as a measurement, or re-raise."
                        ),
                    )
                )
    return findings
