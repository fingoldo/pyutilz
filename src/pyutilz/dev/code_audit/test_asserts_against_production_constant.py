"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a test that checks the code against the code -----------------------------------------------
#
# `assert compute_backoff(3) == BASE_DELAY * 2 ** 3` re-derives the expected value from the same
# constant the implementation reads. Change `BASE_DELAY` and the test moves with it; get the
# formula wrong in both places and the test agrees. It passes for any value of the thing it claims
# to pin, which is the definition of an assertion that cannot fail.
#
# The audited codebase had this in its throttling tests: every expected interval was written as an
# expression over the production constants, so a change that halved the interval everywhere was
# green. The fix is a literal -- if the interval is meant to be 300 seconds, the test says 300, and
# a deliberate change to that number is a deliberate change to the test.
#
# Reported only for an import the test file makes from the code under test, so a constant the test
# defines for itself (a fixture's own `EXPECTED = 300`) is never flagged. Comparisons against a
# CONTAINER of constants, and membership tests, are left alone: `assert result in VALID_STATUSES`
# is checking conformance to a declared set, not re-deriving an answer.

_ASSERT_HELPERS = frozenset({"assertEqual", "assertAlmostEqual", "assertNotEqual", "approx"})


def _imported_from_production(tree: ast.Module) -> set[str]:
    """Names this test module imports from somewhere other than the standard test toolkit."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        root = node.module.split(".")[0]
        if root in {"pytest", "unittest", "typing", "collections", "dataclasses", "pathlib", "os", "sys", "re", "json", "math", "datetime"}:
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            # Constants only. A function or class imported for use is not an expected value.
            if name.isupper() or (name.startswith("_") and name.lstrip("_").isupper()):
                imported.add(name)
    return imported


def _arithmetic_over(node: ast.expr, constants: set[str]) -> set[str]:
    """The production constants this expression does ARITHMETIC on.

    Arithmetic is the whole rule. Any lesser test drowns: reported for every mention of an imported
    upper-case name, this produced 258 hits across three repositories, dominated by four shapes
    that are all correct code --

    * `assert len(tables[INSERT_JOB_SQL]) == 1`, where the constant is a dictionary KEY;
    * `assert len(POLARS_DEFAULT_QUANTILES) == 5`, which pins the constant against a literal and
      is exactly what this rule wants people to write;
    * `assert _ALIASES[alias] == canonical`, a lookup in an imported table;
    * `assert MRMR._FIT_CACHE`, where an all-capitals CLASS name read as a constant.

    None of them re-derive an expected value. `BASE_DELAY * 2 ** 3` does.
    """
    # A CALL anywhere on this side means the constant is feeding the code under test, not standing
    # in for its answer: `choose_corr_backend(CORR_MIN_ROWS - 1, ...) == "numpy"` is a correct
    # boundary test whose expected value is the literal on the other side. That shape was six of
    # sixteen hits.
    if any(isinstance(sub, ast.Call) for sub in ast.walk(node)):
        return set()
    used: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.BinOp):
            continue
        # `CHECKPOINT_DIR / "~01abc.jsonl"` is path construction wearing an operator, not
        # arithmetic over a number.
        if any(isinstance(side, ast.Constant) and isinstance(side.value, str) for side in (sub.left, sub.right)):
            continue
        for operand in (sub.left, sub.right):
            for name in ast.walk(operand):
                if isinstance(name, ast.Name) and name.id in constants:
                    used.add(name.id)
    return used


def _mentions(node: ast.expr, constants: set[str]) -> bool:
    """Whether this expression mentions any of the production constants at all."""
    return any(isinstance(sub, ast.Name) and sub.id in constants for sub in ast.walk(node))


def _asserted_comparisons(tree: ast.Module) -> list[tuple[ast.expr, ast.expr, int]]:
    """(left, right, line) for every equality this module asserts, statement or helper call."""
    pairs: list[tuple[ast.expr, ast.expr, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            compare = node.test
            if len(compare.ops) == 1 and isinstance(compare.ops[0], (ast.Eq, ast.NotEq)):
                pairs.append((compare.left, compare.comparators[0], node.lineno))
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _ASSERT_HELPERS:
                pairs.append((node.args[0], node.args[1], node.lineno))
    return pairs


def scan_test_asserts_against_production_constant(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a test whose expected value is derived from the constant the code under test reads.

    `assert compute_backoff(3) == BASE_DELAY * 2 ** 3` passes for any value of `BASE_DELAY`: change
    the constant and the expectation moves with it, so the assertion cannot fail for the thing it
    claims to pin. An audited codebase had every expected throttle interval written this way, and a
    change that halved them all was green.

    Only imported, upper-case names count, so a constant the test defines for itself is never
    flagged, and only DERIVED uses -- a bare `assert TIMEOUT == config.timeout` is a real claim
    about wiring, not a re-derivation.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        name = py.name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        constants = _imported_from_production(tree)
        if not constants:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for left, right, line in _asserted_comparisons(tree):
            for side, other in ((left, right), (right, left)):
                used = _arithmetic_over(side, constants)
                if not used:
                    continue
                # The other side has to be the thing under test, not a second expression over the
                # same constants -- comparing two derivations of one constant is a different (and
                # rarer) mistake, and reporting it here would blur the message.
                if _mentions(other, constants):
                    continue
                findings.append(
                    Finding(
                        check="test_asserts_against_production_constant",
                        severity="P2",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"the expected value here is derived from `{sorted(used)[0]}`, the "
                            "same constant the code under test reads, so this assertion passes "
                            "for any value of it -- change the constant and the expectation moves "
                            "with it. Write the expected value as a literal, so that changing the "
                            "constant is a deliberate change to this test too."
                        ),
                    )
                )
                break
    return findings
