"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- nan-equality trap --------------------------------------------------


def _is_nan_constant(node: ast.AST) -> bool:
    """True if ``node`` is a literal NaN constant in any of the common forms.

    Catches:
    - ``float("nan")`` / ``float('NaN')``
    - ``np.nan`` / ``numpy.nan`` / ``math.nan``
    - ``float("inf")`` is NOT NaN; not flagged.
    """
    # ``float("nan")`` form
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.strip().lower() == "nan"
    ):
        return True
    # ``np.nan`` / ``numpy.nan`` / ``math.nan`` form
    if isinstance(node, ast.Attribute) and node.attr == "nan":
        return True
    # ``math.inf`` / ``np.inf`` are NOT NaN; let those through.
    return False


def scan_nan_equality(root: Path,
                     exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                     ) -> list[Finding]:
    """Find ``x == NaN`` / ``x != NaN`` shapes.

    NaN doesn't equal anything in IEEE 754 (not even itself), so these
    comparisons ALWAYS yield False. The correct check is
    ``math.isnan(x)`` / ``np.isnan(x)`` / ``pd.isna(x)``.

    Severity: P0. The comparison evaluates False unconditionally, so any
    code-path guarded on detecting NaN never fires.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            # Walk every (op, comparator) pair plus the LHS to cover the
            # 'x == NaN' AND 'NaN == x' AND 'x != NaN' shapes.
            operands = [node.left, *node.comparators]
            ops = node.ops
            for i, op in enumerate(ops):
                if not isinstance(op, (ast.Eq, ast.NotEq)):
                    continue
                lhs = operands[i]
                rhs = operands[i + 1]
                nan_side = None
                if _is_nan_constant(lhs):
                    nan_side = "left"
                elif _is_nan_constant(rhs):
                    nan_side = "right"
                if nan_side is None:
                    continue
                findings.append(Finding(
                    check="nan_equality",
                    severity="P0",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        f"`{type(op).__name__}` comparison against a NaN "
                        f"literal on the {nan_side}. NaN does not equal "
                        f"anything in IEEE 754 -> the result is always "
                        f"False (or True for NotEq). Use `math.isnan(x)` "
                        f"/ `np.isnan(x)` / `pd.isna(x)` instead."
                    ),
                ))
    return findings
