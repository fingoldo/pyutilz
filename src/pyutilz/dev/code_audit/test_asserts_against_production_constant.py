"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

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
# The subset that forms a two-sided comparison by itself. `approx` is NOT one: `pytest.approx(x, rel)`
# takes a tolerance as its second argument, and pairing that with `x` mis-read the tolerance as
# "the other side" -- it is unwrapped by `_unwrap_approx` instead.
_UNITTEST_ASSERT_HELPERS = frozenset({"assertEqual", "assertAlmostEqual", "assertNotEqual"})


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
    # `pytest.approx(BASE_DELAY * 2)` wraps the expected value; bailing on it neutralised the whole
    # check for the (very common) approximate spelling, even though `approx` is already listed as an
    # assertion helper. Unwrap to the value it wraps, then apply the ordinary rule.
    node = _unwrap_approx(node)
    if any(isinstance(sub, ast.Call) for sub in ast.walk(node)):
        return set()
    used: set[str] = set()
    for sub in ast.walk(node):
        # `assert f() == -BASE_DELAY` re-derives the expectation just as `BASE_DELAY * -1` does.
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, (ast.USub, ast.UAdd)) and isinstance(sub.operand, ast.Name) and sub.operand.id in constants:
            used.add(sub.operand.id)
        if not isinstance(sub, ast.BinOp):
            continue
        if _is_path_construction(sub):
            continue
        for operand in (sub.left, sub.right):
            for name in ast.walk(operand):
                if isinstance(name, ast.Name) and name.id in constants:
                    used.add(name.id)
    return used


_PATH_NAME_SUFFIXES = ("DIR", "PATH", "ROOT", "FOLDER")


def _is_path_construction(node: ast.BinOp) -> bool:
    """`CHECKPOINT_DIR / "~01.jsonl"` / `CHECKPOINT_DIR / sub` -- path building, not arithmetic.

    Recognising only a bare string Constant operand missed the two shapes the module's own comment
    names as must-be-silent: an f-string filename and a plain variable segment.
    """
    if any(isinstance(side, ast.Constant) and isinstance(side.value, str) for side in (node.left, node.right)):
        return True
    if not isinstance(node.op, ast.Div):
        return False
    left = node.left
    if isinstance(left, ast.BinOp):
        return _is_path_construction(left)
    if isinstance(left, ast.Name) and left.id.upper().endswith(_PATH_NAME_SUFFIXES):
        return True
    return False


def _unwrap_approx(node: ast.expr) -> ast.expr:
    """`pytest.approx(x)` / `approx(x)` -> `x`; anything else unchanged."""
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "approx":
            return node.args[0]
    return node


def _mentions(node: ast.expr, constants: set[str]) -> bool:
    """Whether this expression mentions any of the production constants at all."""
    return any(isinstance(sub, ast.Name) and sub.id in constants for sub in ast.walk(node))


def _asserted_comparisons(tree: ast.Module) -> list[tuple[ast.expr, ast.expr, int]]:
    """(left, right, line) for every equality this module asserts, statement or helper call."""
    pairs: list[tuple[ast.expr, ast.expr, int]] = []
    # Nodes inside a `test_*` function: the unittest-helper arm applies only there.
    in_test: set[int] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test_"):
            for sub in ast.walk(fn):
                in_test.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            compare = node.test
            if len(compare.ops) == 1 and isinstance(compare.ops[0], (ast.Eq, ast.NotEq)):
                pairs.append((compare.left, compare.comparators[0], node.lineno))
        elif isinstance(node, ast.Call) and len(node.args) >= 2 and id(node) in in_test:
            func = node.func
            # A `self`/`cls` receiver inside a test function: an `assertEqual`-shaped method on an
            # arbitrary object (`recorder.assertEqual(...)`) is somebody else's API, not an assertion.
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}):
                continue
            if func.attr in _UNITTEST_ASSERT_HELPERS:
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
        src_lines = _read_src_lines(py)
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
                # The offending ARGUMENT's own position: a multi-line `self.assertEqual(` reported
                # the opening-paren line, whose snippet showed nothing of the expression at fault.
                offending_line = getattr(side, "lineno", line)
                findings.append(
                    Finding(
                        check="test_asserts_against_production_constant",
                        severity="P2",
                        file=rel,
                        line=offending_line,
                        snippet=_line_text(src_lines, offending_line),
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
