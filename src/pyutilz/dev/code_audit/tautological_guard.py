"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- a threshold ANDed with a pin that makes it vacuous -------------------------------------------
#
# Confirmed instance (Autopsia, 2026-07-26, reason/enumerate.py):
#
#     c.mean >= 0.75 * lead and c is self.causes[0]
#
# `self.causes` is sorted by mean, so the second clause holds only for the element whose mean IS
# `lead` -- and `lead >= 0.75 * lead` is true for any non-negative lead. The leader was therefore
# labelled "strongly supported" unconditionally, and the label carried no information at all. The
# same shape appears wherever a rank/identity pin is ANDed onto a score threshold.
#
# The detector is narrow on purpose: both clauses must concern the SAME target, one must be an
# ordering comparison and the other an identity/equality pin. `x > 0 and x is not None` is not
# flagged (the pin is against None, which does not fix a value); `a > 3 and b is c` is not flagged
# (different targets, a legitimate two-condition guard).

_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _target_key(node: ast.AST) -> str | None:
    """A stable key for the value being tested: `c`, `c.mean`, `row["x"]`. None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _target_key(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Subscript):
        base = _target_key(node.value)
        return f"{base}[]" if base else None
    return None


def _root_of(key: str) -> str:
    """`c.mean` and `c` share the root `c` -- a threshold on an attribute is pinned by pinning the object."""
    return key.split(".")[0].split("[")[0]


def scan_tautological_guards(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``A_threshold and A_pinned_to_one_value`` -- a guard whose second clause makes the first vacuous.

    Flags an ``and`` expression that contains both an ordering comparison (``<``/``<=``/``>``/``>=``)
    on some target and an identity/equality pin (``is``/``==``) on the same target against a
    non-``None``, non-boolean value. Whatever the threshold was meant to express, only one value can
    reach it, so the condition reduces to the pin and the threshold silently stops carrying meaning.

    Severity: P1. The expression evaluates, the branch is taken, and the resulting label or decision
    looks earned. Nothing in a test that only checks the happy path can tell the difference.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And)):
                continue
            thresholded: dict[str, ast.AST] = {}
            pinned: dict[str, ast.AST] = {}
            for operand in node.values:
                if not (isinstance(operand, ast.Compare) and len(operand.ops) == 1):
                    continue
                key = _target_key(operand.left)
                if key is None:
                    continue
                op, right = operand.ops[0], operand.comparators[0]
                if isinstance(op, _ORDERING):
                    thresholded[_root_of(key)] = operand
                elif isinstance(op, (ast.Is, ast.Eq)):
                    # `is None` / `== True` narrow a type, they do not fix a value out of an ordering.
                    if isinstance(right, ast.Constant) and (right.value is None or isinstance(right.value, bool)):
                        continue
                    pinned[_root_of(key)] = operand
            shared = sorted(set(thresholded) & set(pinned))
            for target in shared:
                line = min(thresholded[target].lineno, pinned[target].lineno)  # type: ignore[attr-defined]
                findings.append(
                    Finding(
                        check="tautological_guard",
                        severity="P1",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"`and` combines a threshold on {target!r} with a pin of {target!r} to one value: only that value "
                            f"can reach the threshold, so the threshold is vacuous and the guard reduces to the pin."
                        ),
                    )
                )
    return findings
