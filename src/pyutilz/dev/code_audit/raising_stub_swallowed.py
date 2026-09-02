"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a test stub that raises, into production code that swallows -----------------------------
#
# A test expresses "this must never be called" by raising inside the patched callable. The code
# under test wraps that call in `except Exception` and converts the raise into a benign path, so
# the assertion is swallowed and the test passes whatever happens.
#
# Confirmed instance (upwork scrapers, `test_locate_uses_cache_when_not_stale`): the stub raised
# unconditionally, `_part_has_upwork` caught every exception and returned None ("uncertain"), which
# routed the part into the hit list anyway -- so `hits == cached` held by coincidence, and the
# cache was being re-probed on every run for as long as the test was green.
#
# What makes this worth a mechanical check is that the test LOOKS strict. "Raise if this is
# called" is the strongest way to say "must not be called", and it is exactly the form that a
# broad handler downgrades to nothing.

_BROAD = {"Exception", "BaseException"}

# Patch spellings that install a replacement callable.
_PATCH_FUNCS = {"patch", "setattr", "patch.object"}


def _raises_unconditionally(node: ast.AST) -> bool:
    """Does this function/lambda body do nothing but raise?

    Deliberately strict. A stub that raises only under a condition is expressing "raise in THIS
    case", which is a different and usually legitimate thing -- the degradation-path tests that
    set `side_effect = psycopg2.InterfaceError` precisely so the production handler catches it.
    """
    if isinstance(node, ast.Lambda):
        return False  # a lambda cannot contain a bare `raise` statement
    body = getattr(node, "body", None)
    if not isinstance(body, list) or not body:
        return False
    statements = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    return len(statements) == 1 and isinstance(statements[0], ast.Raise)


def _stub_names_that_raise(scope: ast.AST) -> set[str]:
    """Function names defined WITHIN this scope whose body is a single `raise`.

    Scoped to one test, not to the module, and that is the difference between a working rule
    and a useless one. `_gql` is a name every other test in a file defines for itself: in the
    suite this rule was written against, five tests define a `_gql` that returns a canned
    response and one defines a `_gql` that raises. Collected module-wide, the raising one
    tainted all five, and the rule's single hit on a real repository was a false positive.
    """
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not scope:
            if _raises_unconditionally(node):
                names.add(node.name)
    return names


def _asserts_on_a_raise(func: ast.AST) -> bool:
    """Does this test expect the exception itself -- `pytest.raises`, or an assertion on a return?

    Two exclusions in one. `pytest.raises` means the raise IS the subject. An assertion on the
    driving call's RETURN VALUE means the test is exercising the degradation path on purpose, which
    is the main real false positive for this rule.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name == "raises":
                return True
    return False


def _spy_style_assertions(func: ast.AST) -> bool:
    """Does the test assert on a spy rather than on a value -- `.called`, `.assert_*`, a list?

    This is the shape the rule is about: the test believes the stub's raise is what enforces
    "never called", and asserts something adjacent. If the test asserts on the driving call's
    return value instead, the swallow is the behaviour under test.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute) and (node.attr in ("called", "call_count", "call_args", "call_args_list") or node.attr.startswith("assert_")):
            return True
    return False


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """Does this handler catch Exception/BaseException, or nothing at all?"""
    if handler.type is None:
        return True
    names: list[str] = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
    return bool(set(names) & _BROAD)


def _reraises(handler: ast.ExceptHandler) -> bool:
    """Does this handler re-raise? Then it does not swallow, whatever else it does."""
    return any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def _swallowing_calls(tree: ast.AST) -> set[str]:
    """Names called inside a broad `except` that does not re-raise, in this production module."""
    swallowed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_handler_is_broad(h) and not _reraises(h) for h in node.handlers):
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    target = sub.func
                    name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                    if name:
                        swallowed.add(name)
    return swallowed


def _patched_targets(func: ast.AST, raising_stubs: set[str]) -> list[tuple[str, int]]:
    """(patched attribute name, line) for every patch installing one of *raising_stubs*."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        fname = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if fname not in {"patch", "setattr", "object"}:
            continue

        replacement_is_a_raiser = False
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Name) and arg.id in raising_stubs:
                replacement_is_a_raiser = True
        if not replacement_is_a_raiser:
            continue

        attr = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                attr = arg.value.rsplit(".", 1)[-1]
        if attr is None:
            for arg in node.args:
                if isinstance(arg, ast.Attribute):
                    attr = arg.attr
        if attr:
            out.append((attr, node.lineno))
    return out


def scan_raising_stub_swallowed(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find test stubs that raise into production code which catches broadly and carries on.

    The test says "this must never be called" in the strongest available way -- by raising -- and
    a broad handler downstream turns that into a benign path, so the assertion beside it cannot
    fail. That is how a cache stayed unused, and re-probed on every run, behind a green test.

    Reported only when the test does NOT expect the exception (`pytest.raises`) and DOES assert on
    a spy rather than on the driving call's return value. A test that asserts on the return value
    is exercising the degradation path on purpose, which is the rule's main false positive.
    """
    findings: list[Finding] = []

    production_swallows: dict[str, set[str]] = {}
    test_files: list[Path] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        if py.name.startswith("test_") or "tests" in py.parts:
            test_files.append(py)
        else:
            swallowed = _swallowing_calls(tree)
            if swallowed:
                production_swallows[rel] = swallowed

    if not production_swallows:
        return findings

    for py in test_files:
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not func.name.startswith("test_"):
                continue
            raising_stubs = _stub_names_that_raise(func)
            if not raising_stubs:
                continue
            if _asserts_on_a_raise(func) or not _spy_style_assertions(func):
                continue
            for attr, line in _patched_targets(func, raising_stubs):
                swallowers = sorted(f for f, names in production_swallows.items() if attr in names)
                if not swallowers:
                    continue
                findings.append(
                    Finding(
                        check="raising_stub_swallowed",
                        severity="P2",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"`{func.name}` patches `{attr}` with a stub that raises unconditionally, "
                            f"but {swallowers[0]} calls `{attr}` inside a broad `except` that does not "
                            "re-raise -- the raise is converted to a benign path and the assertion "
                            "beside it cannot fail. Assert on a RECORDED call instead of raising."
                        ),
                    )
                )
    return findings
