"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- test functions that cannot fail -------------------------------------
#
# Twenty-seven findings in the 2026-09-01 mlframe audit, every one a test that passed for a reason unrelated to
# the property it claimed to check, several provably incapable of failing at all:
#
# * A negative contract ("bf16 must NOT be auto-set on CPU") whose whole body was a `fit()` call, closing with
#   "if we got here without crashing, the plumbing worked" -- false, because the library under test ACCEPTS the
#   wrong value with a warning rather than an error.
# * A Hypothesis property test wrapping all four of its assertions in `except AssertionError: pass`, so across
#   50 generated examples none could fail the suite.
# * Five prediction-sanity tests guarded by `hasattr` + not-None + non-empty, so a suite that stopped producing
#   predictions -- the regression they are the last line of defence against -- passed all five by satisfying
#   none of the guards.
# * A documented expectation written as an `if` with a `pass` body, directly beneath the comment saying the
#   state it permits would be wrong.
# * An imperative `pytest.xfail(...)` discarding the measurement just taken -- one of which was concealing a
#   gap that had already CLOSED.
#
# Complementary to `vacuous_assertion`, which judges the assertion EXPRESSION (a full-domain `or`). This one
# judges STRUCTURE: whether any assertion is reachable and unconditional at all.

DEFAULT_TEST_PREFIX = "test_"
_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# Calls that carry a real check even though they are not `assert` statements.
DEFAULT_ASSERTING_CALLS: frozenset[str] = frozenset(
    {"raises", "warns", "approx", "fail", "assert_array_equal", "assert_allclose", "assert_frame_equal", "assert_series_equal"}
)

_REASON_HELP = {
    "no-assert": "the body runs code and checks nothing; 'it did not crash' is not the contract, and a library that accepts a wrong value with a warning makes that reasoning false outright",
    "swallows-assertionerror": "`except AssertionError: pass` (or a broad except with no re-raise) means no assertion inside can ever fail the suite",
    "all-asserts-conditional": "every assertion sits behind an `if`, so the STRONGER failure -- the object not being produced at all -- skips the check instead of failing it; assert the precondition too",
    "pass-body-if": "a documented expectation written as a no-op; if the comment says a state is wrong, assert that it does not occur",
    "imperative-xfail": "`pytest.xfail(...)` discards the measurement just taken; measure first and xfail only when the gap is confirmed still open, so a gap that CLOSES is reported rather than concealed",
}


def _own_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """Walk ``func`` without descending into nested function definitions -- a helper's assertions are its own."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _FUNC_NODES):
            stack.extend(ast.iter_child_nodes(node))


def _has_any_check(func: ast.AST, asserting_calls: frozenset[str]) -> bool:
    """True when the function contains an assert statement or a call that performs a check."""
    for node in _own_nodes(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if name in asserting_calls or name.startswith("assert"):
                return True
    return False


def _swallows_assertion_error(func: ast.AST) -> bool:
    """True when a handler catches AssertionError (directly or via Exception/bare) and does not re-raise."""
    for node in _own_nodes(func):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = {"BaseException"} if node.type is None else {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
        if not (caught & {"AssertionError", "Exception", "BaseException"}):
            continue
        if any(isinstance(inner, ast.Raise) for inner in ast.walk(node)):
            continue
        # A handler containing its own assertion is CHECKING the failure, not swallowing it.
        if any(isinstance(inner, ast.Assert) for inner in ast.walk(node)):
            continue
        return True
    return False


def _every_assert_is_conditional(func: ast.AST) -> bool:
    """True when the function has assertions and every one of them is nested inside an `if`."""
    asserts = [n for n in _own_nodes(func) if isinstance(n, ast.Assert)]
    if not asserts:
        return False
    conditional: set[int] = set()
    for node in _own_nodes(func):
        if isinstance(node, ast.If):
            conditional.update(id(inner) for inner in ast.walk(node) if isinstance(inner, ast.Assert))
    return all(id(a) in conditional for a in asserts)


def _has_pass_only_if(func: ast.AST) -> bool:
    """True when an `if` body is exactly `pass` -- a documented expectation written as a no-op."""
    return any(isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass) for node in _own_nodes(func))


def _has_imperative_xfail(func: ast.AST) -> bool:
    """True when the body calls ``pytest.xfail(...)``, which discards whatever was measured."""
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "xfail" for node in _own_nodes(func))


def _reasons(func: ast.AST, asserting_calls: frozenset[str]) -> list[str]:
    """Every nondiscriminating shape this function exhibits, as short slugs."""
    out: list[str] = []
    if not _has_any_check(func, asserting_calls):
        out.append("no-assert")
    if _swallows_assertion_error(func):
        out.append("swallows-assertionerror")
    if _every_assert_is_conditional(func):
        out.append("all-asserts-conditional")
    if _has_pass_only_if(func):
        out.append("pass-body-if")
    if _has_imperative_xfail(func):
        out.append("imperative-xfail")
    return out


def scan_nondiscriminating_test_functions(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    *,
    test_prefix: str = DEFAULT_TEST_PREFIX,
    asserting_calls: frozenset[str] = DEFAULT_ASSERTING_CALLS,
) -> list[Finding]:
    """Flag a test function that cannot fail for the reason it claims to check.

    Considers only files named ``<test_prefix>*.py`` and functions named ``<test_prefix>*``, so fixtures,
    helpers and conftest scaffolding are out of scope by construction. Reports every shape a function exhibits,
    not just the first.

    Severity: P1. Such a test is worse than absent -- it reads as coverage of the property it names while
    passing unconditionally.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not py.name.startswith(test_prefix):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, _FUNC_NODES) or not func.name.startswith(test_prefix):
                continue
            reasons = _reasons(func, asserting_calls)
            if not reasons:
                continue
            findings.append(
                Finding(
                    check="nondiscriminating_test",
                    severity="P1",
                    file=rel,
                    line=func.lineno,
                    snippet=_line_text(src_lines, func.lineno),
                    detail=f"`{func.name}` cannot fail for the reason it claims to check -- " + "; ".join(f"{r}: {_REASON_HELP[r]}" for r in reasons),
                )
            )
    return findings
