"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- one sibling missing the guard all the others open with -----------------------------------
#
# Five methods on a class start `if self._closed: return`. The sixth does not, and it is the one
# that runs against a closed handle. The guard is not documented anywhere; it is documented by
# being repeated, and a repetition with one hole in it looks exactly like a repetition.
#
# This is the cheapest defect class there is to introduce -- a method added later, by someone
# reading the two siblings that happened not to need the guard -- and among the most expensive to
# find, because nothing about the new method looks wrong in isolation.
#
# The rule needs the majority to be unambiguous:
#
# * at least THREE siblings open with the same guard, written the same way. Two is a coincidence;
#   three is a convention.
# * the unguarded sibling must touch the RESOURCE the guarded ones touch -- not the name the guard
#   tests. That distinction is the rule. The whole point of the defect is that the unguarded
#   method never mentions `_closed`; requiring it to do so, as the first version of this did, made
#   the rule silent on its own worked example.
# * the unguarded sibling must not check the guarded name anywhere itself. A guard spelled
#   differently, or placed later in the body, is still a guard, and reporting it would be telling
#   the author to write the check they already wrote.
#
# Dunder methods are excluded on both sides: `__repr__` and `__eq__` legitimately run on a closed
# or half-built object, and that is what they are for.

# `self` and `cls` are mentioned by almost every statement in a method, so letting either into the
# set of things the guard watches makes "this sibling uses what the guard tests" true for all of
# them. The first run of this rule reported a method as using `_closed` when it uses only `self`
# and `_fd`; the same vacuous-token trap cost effect_flag_outside_its_effect a false positive too.
_VACUOUS_TOKENS = frozenset({"self", "cls"})

_MIN_SIBLINGS = 3


def _leading_guard(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ast.stmt] | None:
    """The early-exit guard this function opens with, as normalised text."""
    body = [stmt for stmt in func.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
    if not body or not isinstance(body[0], ast.If) or body[0].orelse:
        return None
    guard = body[0]
    if not all(isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break, ast.Pass)) for stmt in guard.body):
        return None
    return ast.dump(guard.test), guard


def _names_in(node: ast.AST) -> set[str]:
    """Names and attribute names mentioned anywhere in this subtree."""
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
    return found


def _tests_any_of(func: ast.AST, names: set[str]) -> bool:
    """Whether this function checks any of those names in a condition of its own.

    A guard spelled differently, or placed later in the body, is still a guard -- reporting it
    would be telling the author to write the check they already wrote.
    """
    for sub in ast.walk(func):
        test = None
        if isinstance(sub, (ast.If, ast.IfExp, ast.While)):
            test = sub.test
        elif isinstance(sub, ast.Assert):
            test = sub.test
        if test is not None and _names_in(test) & names:
            return True
    return False


def _self_attributes(node: ast.AST) -> set[str]:
    """Attributes reached through `self` in this subtree -- the object's own state.

    Any wider notion of "what this method touches" is worthless here. Intersecting all names
    produced resources like `['os', 'path']` and `['Any', 'Dict', 'Optional', 'ValueError']`:
    imported type names and builtins appear in every method of a class, so the intersection is
    never empty and the link it forms means nothing. A guard on `self._closed` protects the
    object's state, so the object's state is what the link has to be made of.
    """
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in ("self", "cls"):
            found.add(sub.attr)
    return found


def _shared_resource(guarded: list[ast.FunctionDef | ast.AsyncFunctionDef], watched: set[str]) -> set[str]:
    """What every guarded sibling touches, besides the name the guard itself tests.

    This is the link between the guard and the sibling that lacks it: the guarded methods all use
    `self._fd`, so a sibling that also uses `self._fd` and skips the guard is the defect. Matching
    on the guard's own name instead finds nothing, because not mentioning it is the whole problem.
    """
    if not guarded:
        return set()
    shared = _self_attributes(guarded[0])
    for method in guarded[1:]:
        shared &= _self_attributes(method)
    return shared - watched - _VACUOUS_TOKENS


def scan_sibling_guard_missing(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a method missing the early-exit guard its siblings all open with.

    Five methods start `if self._closed: return` and the sixth does not -- and the sixth is the one
    that runs against a closed handle. The guard is documented only by being repeated, and a
    repetition with one hole in it reads exactly like a repetition.

    At least three siblings must share the guard, written the same way: two is a coincidence, and
    at a threshold of two this reported forty-odd pairs across four codebases, mostly a method
    validating an argument beside one that does not take it. The unguarded sibling must also use
    what the guard tests, and must not check that name anywhere itself.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for klass in ast.walk(tree):
            if not isinstance(klass, ast.ClassDef):
                continue
            methods = [node for node in klass.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("__")]
            if len(methods) <= _MIN_SIBLINGS:
                continue

            guards: dict[str, list[ast.stmt]] = {}
            for method in methods:
                found = _leading_guard(method)
                if found is not None:
                    guards.setdefault(found[0], []).append(found[1])

            for occurrences in guards.values():
                if len(occurrences) < _MIN_SIBLINGS:
                    continue
                guarded_ids = {id(occurrence) for occurrence in occurrences}
                watched = _names_in(occurrences[0]) - _VACUOUS_TOKENS
                guarded_methods = [m for m in methods if (g := _leading_guard(m)) and id(g[1]) in guarded_ids]
                resource = _shared_resource(guarded_methods, watched)
                if not resource:
                    continue
                for method in methods:
                    found = _leading_guard(method)
                    if found is not None and id(found[1]) in guarded_ids:
                        continue
                    if not (_self_attributes(method) & resource) or _tests_any_of(method, watched):
                        continue
                    guard_text = _line_text(src_lines, occurrences[0].lineno)
                    findings.append(
                        Finding(
                            check="sibling_guard_missing",
                            severity="P2",
                            file=rel,
                            line=method.lineno,
                            snippet=_line_text(src_lines, method.lineno),
                            detail=(
                                f"`{klass.name}.{method.name}` touches {sorted(resource)} but does not "
                                f"open with `{guard_text}`, which {len(occurrences)} of its "
                                "siblings do and nothing else documents. A convention recorded "
                                "only by repetition reads the same with one hole in it as without."
                            ),
                        )
                    )
                    break
    return findings
