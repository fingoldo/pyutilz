"""Meta-test: every registered code_audit scanner has BOTH a positive and a negative test.

``tests/test_meta/test_test_source_parity.py`` exempts roughly fifty ``code_audit`` scanner
modules from its own per-module test requirement on the grounds that "every scanner submodule is
exercised jointly by tests/code_audit/ (positive+negative case per scanner)". That claim was
prose only: nothing checked it, and the 2026-09-03 test-suite audit found it false for
``partial_guard_across_siblings``, which had two tests that both asserted a finding and none that
asserted silence -- so any widening of the rule could have started flagging clean families
without a single test going red.

What "positive" and "negative" mean here:

* POSITIVE -- some test asserts the scanner returned a NON-empty result for a constructed input:
  ``len(findings) == 1``, ``len(findings) >= 1``, ``findings[0].check``, ``assert findings``,
  ``[f.name for f in findings] == ["x"]``.
* NEGATIVE -- some test asserts an EMPTY result: ``findings == []``, ``len(findings) == 0``,
  ``assert not findings``.

Both are read statically out of the test sources rather than by running anything, so the gate
costs one AST parse per test module and cannot be satisfied by a test that merely calls the
scanner and throws the result away. Polarity is attributed PER SCANNER: a test that exercises two
scanners contributes only the assertions that name the scanner's own result.

The result set of a scanner is followed through assignment, so
``subset = [f for f in findings if ...]; assert len(subset) == 1`` counts, and
``functools.partial``-wrapped registry entries are unwrapped to the underlying function.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import Dict, List, Set, Tuple

from pyutilz.dev.code_audit import get_scanners

TESTS_DIR = Path(__file__).resolve().parent.parent

# A scanner may be listed here only when a CLEAN input genuinely cannot exist for it -- i.e. the
# scanner reports on something other than a defect in the scanned tree. Nothing qualifies today:
# every one of the 95 registered scanners answers "is this shape present?", so an input without
# the shape is always constructible. An entry needs the specific reason written out, not a
# pointer to a test that is merely missing; "hard to construct" is a reason to write the test.
_NO_CLEAN_CASE_POSSIBLE: Dict[str, str] = {}


def _registered_scanner_function_names() -> Dict[str, List[str]]:
    """Underlying function ``__name__`` -> the registry key(s) it is registered under."""
    out: Dict[str, List[str]] = {}
    for name, fn in get_scanners().items():
        while isinstance(fn, functools.partial):
            fn = fn.func
        out.setdefault(getattr(fn, "__name__", ""), []).append(name)
    out.pop("", None)
    return out


def _scanner_aliases(tree: ast.Module, known: Dict[str, List[str]]) -> Dict[str, str]:
    """Local identifier -> scanner function name, over every import in the module.

    Walks the whole tree rather than ``tree.body`` because several suites import a scanner
    inside the test function that uses it.
    """
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "code_audit" in node.module:
            for alias in node.names:
                if alias.name in known:
                    out[alias.asname or alias.name] = alias.name
    return out


def _int_value(node: ast.expr):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_value(node.operand)
        return None if inner is None else -inner
    return None


def _result_names(func: ast.AST, aliases: Set[str]) -> Set[str]:
    """Local names holding the scanner's result, following `subset = [f for f in findings if ...]`."""
    result_names: Set[str] = set()
    for _ in range(3):  # a few hops is enough for the chains the suites actually use
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(x, ast.Name) and (x.id in aliases or x.id in result_names) for x in ast.walk(node.value)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result_names.add(target.id)
    return result_names


def _compared_polarity(test: ast.expr, is_result) -> Tuple[bool, bool, bool]:
    """``(positive, negative, compared)`` from the comparisons in one assert's test expression."""
    positive = negative = compared = False
    for cmp_node in [x for x in ast.walk(test) if isinstance(x, ast.Compare)]:
        left = cmp_node.left
        counts = isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len" and is_result(left)
        for op, comparator in zip(cmp_node.ops, cmp_node.comparators):
            if counts:
                value = _int_value(comparator)
                if value is None:
                    continue
                compared = True
                if isinstance(op, ast.Eq):
                    negative = negative or value == 0
                    positive = positive or value > 0
                elif isinstance(op, (ast.Gt, ast.GtE)):
                    positive = True
            elif is_result(left) and isinstance(op, ast.Eq) and isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
                compared = True
                positive = positive or bool(comparator.elts)
                negative = negative or not comparator.elts
    return positive, negative, compared


def _bare_polarity(test: ast.expr, is_result) -> Tuple[bool, bool]:
    """``(positive, negative)`` for an assert carrying no comparison: `assert findings` / `assert not findings`."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and is_result(test.operand):
        return False, True
    if isinstance(test, (ast.Name, ast.Subscript, ast.Attribute)) and is_result(test):
        return True, False
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id in ("any", "all") and is_result(test):
        return True, False
    return False, False


def _polarity(func: ast.AST, aliases: Set[str]) -> Tuple[bool, bool]:
    """``(asserts_non_empty, asserts_empty)`` for one test function and one scanner."""
    result_names = _result_names(func, aliases)

    def is_result(node: ast.AST) -> bool:
        return any(isinstance(x, ast.Name) and (x.id in aliases or x.id in result_names) for x in ast.walk(node))

    positive = negative = False
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Assert) or not is_result(stmt.test):
            continue
        cmp_pos, cmp_neg, compared = _compared_polarity(stmt.test, is_result)
        positive, negative = positive or cmp_pos, negative or cmp_neg
        if compared:
            continue
        bare_pos, bare_neg = _bare_polarity(stmt.test, is_result)
        positive, negative = positive or bare_pos, negative or bare_neg
    return positive, negative


def _coverage() -> Dict[str, Dict[str, List[str]]]:
    known = _registered_scanner_function_names()
    coverage: Dict[str, Dict[str, List[str]]] = {name: {"positive": [], "negative": []} for name in get_scanners()}
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _scanner_aliases(tree, known)
        if not aliases:
            continue
        rel = path.relative_to(TESTS_DIR).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or not func.name.startswith("test_"):
                continue
            called = {aliases[x.func.id] for x in ast.walk(func) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in aliases}
            for function_name in called:
                positive, negative = _polarity(func, {a for a, f in aliases.items() if f == function_name})
                for registry_name in known[function_name]:
                    if positive:
                        coverage[registry_name]["positive"].append(rel + "::" + func.name)
                    if negative:
                        coverage[registry_name]["negative"].append(rel + "::" + func.name)
    return coverage


def test_every_registered_scanner_has_a_positive_case():
    """A scanner with no test that makes it FIRE is a scanner nobody has seen work."""
    coverage = _coverage()
    missing = sorted(name for name, hits in coverage.items() if not hits["positive"])
    assert not missing, (
        f"{len(missing)} registered scanner(s) have no test asserting a non-empty result. Add a "
        f"constructed input carrying the defect and assert the finding:\n  " + "\n  ".join(missing)
    )


def test_every_registered_scanner_has_a_negative_case():
    """A scanner with no clean-input test is a scanner whose precision nothing pins: it could
    match everything and the suite would stay green."""
    coverage = _coverage()
    missing = sorted(name for name, hits in coverage.items() if not hits["negative"] and name not in _NO_CLEAN_CASE_POSSIBLE)
    assert not missing, (
        f"{len(missing)} registered scanner(s) have no test asserting an EMPTY result. Add a clean "
        f"input close enough to the positive case that an over-matching scanner would flag it, or "
        f"add the scanner to _NO_CLEAN_CASE_POSSIBLE with a written reason:\n  " + "\n  ".join(missing)
    )


def test_no_stale_clean_case_exemptions():
    """An exemption that names a scanner which is no longer registered, or which now HAS a
    negative test, is dead prose that would silently excuse a future regression."""
    coverage = _coverage()
    stale = sorted(name for name in _NO_CLEAN_CASE_POSSIBLE if name not in coverage or coverage[name]["negative"])
    assert not stale, "remove these obsolete _NO_CLEAN_CASE_POSSIBLE entries: " + ", ".join(stale)
