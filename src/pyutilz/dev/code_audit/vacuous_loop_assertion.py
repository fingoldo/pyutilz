"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a test whose every assertion is inside a loop that may not run ----------------------------
#
# `for row in rows: assert row.uid` proves nothing when `rows` is empty, and the test passes. It is
# the most comfortable way for a test to stop testing: the query changes, the fixture stops
# matching, the collection comes back empty, and the suite stays green while the assertion inside
# has not executed once in months.
#
# The fix is one line -- `assert rows` before the loop -- which is also exactly what makes this
# statically decidable: a test that says how many it expects cannot silently drop to zero.
#
# Reported only when the test has NOTHING else holding it up:
#
# * every assertion in the function is inside a `for`. That one condition covers the fix as well:
#   `assert rows` before the loop IS an assertion outside it, so a test that pins the count is
#   already silent. An earlier draft had a separate "does anything guard the collection" check;
#   it could never fire, because anything it would have matched the outside-assertion rule had
#   already let through.
# * the iterated collection is not a literal (`for x in [1, 2, 3]` cannot be empty), and not a
#   `parametrize` argument, which pytest has already counted.
# * no `pytest.raises`, `assert_called`, mock assertion or `pytest.fail` outside the loop -- any of
#   those is an assertion the loop does not own.

_ASSERT_CALLS = ("assert_called", "assert_not_called", "assert_has_calls", "assert_any_call", "assert_awaited", "fail", "raises")


def _assertion_nodes(func: ast.AST) -> list[ast.AST]:
    """Every statement that asserts something: a bare `assert`, a mock check, `pytest.raises`."""
    found: list[ast.AST] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            found.append(node)
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if any(name.startswith(prefix) or name == prefix for prefix in _ASSERT_CALLS):
                found.append(node)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
                    if name in {"raises", "warns"}:
                        found.append(node)
    return found


def _nonempty_literal_names(scopes: list[ast.AST]) -> set[str]:
    """Names bound to a non-empty literal collection in any of these scopes.

    This is what separates a collection that could come back empty from one that cannot. Without
    it the rule reported 478 loops across four codebases, dominated by iteration over module-level
    registries and constants -- `_PROVIDER_MODULES.items()`, `_CC_REGISTRY.items()`,
    `_KNOWN_STUB_ONLY_FILES` -- none of which can be empty and none of which is a defect.
    """
    bound: set[str] = set()
    for scope in scopes:
        for node in ast.walk(scope):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is None or not _is_nonempty_literal(value):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return bound


def _is_nonempty_literal(node: ast.expr) -> bool:
    """Whether this expression is a collection literal with something in it."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return False


def _cannot_be_empty(node: ast.expr, known: set[str]) -> bool:
    """Whether this iterable is provably non-empty by construction."""
    if _is_nonempty_literal(node):
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.Call):
        func = node.func
        # `REGISTRY.items()` is as non-empty as `REGISTRY`.
        if isinstance(func, ast.Attribute) and func.attr in {"items", "keys", "values"}:
            return _cannot_be_empty(func.value, known)
        #  in full, not just .  is an INDEX loop:
        # what could be empty is the frame, decided by a fixture this cannot see, and it supplied
        # the bulk of the remaining hits in one codebase. The shape this rule is for is iterating
        # a RESULT SET, where emptiness is the thing the reader can act on.
        if isinstance(func, ast.Name) and func.id == "range":
            return True
        if isinstance(func, ast.Name) and func.id in {"enumerate", "sorted", "list", "tuple", "set", "reversed"} and node.args:
            return _cannot_be_empty(node.args[0], known)
        if isinstance(func, ast.Name) and func.id == "zip" and node.args:
            return all(_cannot_be_empty(arg, known) for arg in node.args)
    if isinstance(node, ast.Attribute):
        return node.attr in known
    return False


def scan_vacuous_loop_assertion(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a test whose every assertion sits inside a loop that may iterate zero times.

    `for row in rows: assert row.uid` proves nothing when `rows` is empty, and the test passes. It
    is the most comfortable way for a test to stop testing: the query changes, the fixture stops
    matching, the collection comes back empty, and the suite stays green while the assertion has
    not run in months.

    Silent when anything asserts the collection is non-empty first, when the iterable is a literal
    that cannot be empty, and when any assertion sits outside the loop.
    """
    findings: list[Finding] = []

    # PACKAGE-WIDE, because a test iterates registries it imports: `_ALIASES.items()`,
    # `_PROVIDER_MODULES.items()`, `_CC_REGISTRY.items()`, `EXPECTED_ABSENT.items()`. Resolved only
    # within the test file, those read as collections that might be empty, and they were the bulk
    # of this rule's remaining 176 hits across four codebases.
    everywhere: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is not None:
            everywhere |= _nonempty_literal_names([tree])

    for py in _iter_py_files(root, exclude_dirs):
        # No filename filter: `_is_test` already requires a `test_`-prefixed function, and adding a
        # path check on top of it was dead code that read like protection.
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        module_constants = everywhere | _nonempty_literal_names([tree])
        for func in ast.walk(tree):
            # Inlined rather than a helper returning bool: mypy needs the isinstance here to
            # narrow `func`, and a TypeGuard import only for that was dead weight vulture
            # was right to flag.
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or not func.name.startswith("test_"):
                continue
            known = module_constants | _nonempty_literal_names([func])
            assertions = _assertion_nodes(func)
            if not assertions:
                continue
            loops = [node for node in ast.walk(func) if isinstance(node, (ast.For, ast.AsyncFor))]
            if not loops:
                continue

            inside: set[int] = set()
            for loop in loops:
                inside |= {id(sub) for sub in ast.walk(loop)} - {id(loop)}
            if any(id(node) not in inside for node in assertions):
                continue

            for loop in loops:
                owned = [node for node in assertions if id(node) in {id(sub) for sub in ast.walk(loop)}]
                if not owned or _cannot_be_empty(loop.iter, known):
                    continue
                # Only a collection PRODUCED here: a call, or a comprehension. Iterating a name
                # handed in by a fixture or imported from a registry is where every one of this
                # rule's 344 remaining hits lived, and its emptiness is decided somewhere this
                # cannot see. A query result can be empty in a way the reader can act on.
                if not isinstance(loop.iter, (ast.Call, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
                    continue
                findings.append(
                    Finding(
                        check="vacuous_loop_assertion",
                        severity="P2",
                        file=rel,
                        line=loop.lineno,
                        snippet=_line_text(src_lines, loop.lineno),
                        detail=(
                            f"every assertion in `{func.name}` is inside this loop, and nothing "
                            "establishes that it iterates at all. If the collection comes back "
                            "empty the test passes having asserted nothing, and it will keep "
                            "passing. One line before the loop -- `assert <collection>`, or a "
                            "count -- is the whole fix."
                        ),
                    )
                )
                break
    return findings
