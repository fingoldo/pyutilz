"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- vacuous / tautological test assertions -------------------------------
#
# Class of bug confirmed repeatedly in the 2026-07-21 test-suite-quality audit:
# `assert result is None or isinstance(result, dict)` when the function's only possible outcomes
# ARE `None` or a dict (tests/test_serialization.py); `assert len(d) > 0 or d == {}` -- a logical
# tautology for a dict, since `len(d)==0` and `d=={}` are equivalent (tests/test_web.py); a bare
# `assert True` after a block that would already have failed via pytest.fail in its own except
# clause (tests/test_numbalib.py). These pass regardless of what the tested code actually does,
# so a real regression sails through silently.


def _looks_like_full_domain_or(node: ast.BoolOp) -> bool:
    """Heuristic: an ``A or B`` (or longer) boolop where every operand is a comparison of the
    SAME left-hand target against ``None``/an empty-collection-literal, or an ``isinstance(X, T)``
    call on that same target -- i.e. the assertion accepts "None, empty, or any instance of T",
    which is every value that target's static type could plausibly hold. Conservative: only
    flags when there are >=2 operands and every operand matches one of the known-vacuous shapes.
    """
    if not isinstance(node.op, ast.Or) or len(node.values) < 2:
        return False
    targets: set[str] = set()
    shapes = 0
    for operand in node.values:
        # X is None / X == {} / X == [] / X == ""
        if isinstance(operand, ast.Compare) and len(operand.ops) == 1:
            left = operand.left
            op = operand.ops[0]
            right = operand.comparators[0]
            target_src = ast.dump(left)
            if isinstance(op, ast.Is) and isinstance(right, ast.Constant) and right.value is None:
                targets.add(target_src)
                shapes += 1
                continue
            if isinstance(op, ast.Eq) and isinstance(right, (ast.Dict, ast.List, ast.Constant)):
                if isinstance(right, ast.Dict) and right.keys:
                    continue  # non-empty literal -- not a vacuous "any empty" shape
                if isinstance(right, ast.List) and right.elts:
                    continue
                if isinstance(right, ast.Constant) and right.value not in (None, "", 0):
                    continue
                targets.add(target_src)
                shapes += 1
                continue
        # isinstance(X, T)
        if isinstance(operand, ast.Call) and isinstance(operand.func, ast.Name) and operand.func.id == "isinstance" and len(operand.args) == 2:
            targets.add(ast.dump(operand.args[0]))
            shapes += 1
            continue
        # len(X) > 0  (paired with X == {} elsewhere -- tautology for any Sized)
        if (
            isinstance(operand, ast.Compare)
            and len(operand.ops) == 1
            and isinstance(operand.ops[0], (ast.Gt, ast.GtE))
            and isinstance(operand.left, ast.Call)
            and isinstance(operand.left.func, ast.Name)
            and operand.left.func.id == "len"
            and operand.left.args
        ):
            targets.add(ast.dump(operand.left.args[0]))
            shapes += 1
            continue
        return False
    # All operands matched a known-vacuous shape AND they all refer to the same target
    # (otherwise it's a legitimate multi-variable check, not a single-value "accept anything").
    return shapes == len(node.values) and len(targets) == 1


def scan_vacuous_assertions(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    test_glob: str = "test_*.py",
) -> list[Finding]:
    """Find `assert` statements in test files whose condition is true regardless of what the
    tested code actually does: a same-target ``A or B or ...`` boolop where every operand is a
    known-vacuous shape (``X is None``, ``X == {}``/``[]``/``""``, ``isinstance(X, T)``,
    ``len(X) > 0`` alongside ``X == {}``), or a bare ``assert True``/``assert 1``.

    Severity: P2 (the test runs and "passes" but verifies nothing about the code under test --
    a real regression in that code path goes undetected).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not py.name.startswith("test_") and test_glob == "test_*.py":
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            test = node.test
            if isinstance(test, ast.Constant) and test.value is True:
                findings.append(Finding(
                    check="vacuous_assertion",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail="`assert True` -- verifies nothing; if this follows a try/except that already pytest.fail()s on error, the assert is dead weight.",
                ))
            elif isinstance(test, ast.BoolOp) and _looks_like_full_domain_or(test):
                findings.append(Finding(
                    check="vacuous_assertion",
                    severity="P2",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        "`A or B or ...` where every operand is a known-vacuous shape (is None / "
                        "== empty-literal / isinstance(X, T) / len(X)>0) on the SAME target -- "
                        "this accepts nearly any value the target could hold, so the assertion "
                        "passes regardless of what the tested code returns. Assert the actual "
                        "expected value instead."
                    ),
                ))
    return findings
