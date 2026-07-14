"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- duplicate conditions (copy-paste bugs) ----------------------------------
#
# ``x.endswith('ssions') or x.endswith('ssions')`` -- the same operand twice
# in one BoolOp -- is almost always a copy-paste typo where the second operand
# was meant to be a DIFFERENT value (a second suffix, a second key, a second
# flag). The condition still parses and runs, so the intended second check is
# silently never performed. Same story for an ``elif`` whose test is identical
# to a preceding branch's test: the later branch is unreachable dead code.
#
# Both shapes were confirmed in the wild during a 2026-07 large-scale triage
# of a downstream codebase (a French-morphology audit script checked the
# identical suffix twice, so its second subjunctive ending was never audited).


def _operand_key(node: ast.AST) -> str:
    """Structural fingerprint of an expression, ignoring positions.

    ``ast.dump`` without attributes is position-independent, so two
    textually identical operands on one line compare equal.
    """
    return ast.dump(node)


def scan_duplicate_conditions(root: Path,
                              exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
                              ) -> list[Finding]:
    """Find structurally identical repeated conditions:

    1. The same operand appearing twice in one ``and``/``or`` BoolOp
       (``a == 1 or a == 1``) -- the duplicate contributes nothing, and the
       author almost certainly meant a different second value.
    2. An ``elif`` whose test is identical to the test of a preceding
       branch in the same if/elif chain -- the later branch is dead.

    Severity: P2 (the code runs, but an intended check is silently missing
    or an intended branch is silently unreachable).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        # ast.walk visits every If, including the nested Ifs that ARE the
        # elif branches of an outer chain. Starting a chain walk at one of
        # those would re-report the same duplicate once per preceding
        # branch. Only start chains at chain HEADS (Ifs that are not
        # themselves the sole If inside another If's orelse).
        elif_nodes: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.If) and len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If):
                elif_nodes.add(id(n.orelse[0]))
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                seen: dict[str, ast.AST] = {}
                for operand in node.values:
                    key = _operand_key(operand)
                    if key in seen:
                        op_word = "and" if isinstance(node.op, ast.And) else "or"
                        findings.append(Finding(
                            check="duplicate_condition",
                            severity="P2",
                            file=rel,
                            line=operand.lineno,
                            snippet=_line_text(src_lines, operand.lineno),
                            detail=(
                                f"identical operand repeated in one `{op_word}` "
                                f"expression -- likely a copy-paste typo where the "
                                f"second operand was meant to be a different value; "
                                f"the intended second check is silently never performed."
                            ),
                        ))
                    else:
                        seen[key] = operand
            elif isinstance(node, ast.If) and id(node) not in elif_nodes:
                # Walk the elif chain: If.orelse == [If(...)] is an elif.
                chain_tests: dict[str, int] = {_operand_key(node.test): node.test.lineno}
                current = node
                while len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                    key = _operand_key(current.test)
                    if key in chain_tests:
                        findings.append(Finding(
                            check="duplicate_condition",
                            severity="P2",
                            file=rel,
                            line=current.test.lineno,
                            snippet=_line_text(src_lines, current.test.lineno),
                            detail=(
                                f"elif test is identical to the branch at line "
                                f"{chain_tests[key]} in the same if/elif chain -- "
                                f"this branch is unreachable dead code (the earlier "
                                f"branch always wins)."
                            ),
                        ))
                    else:
                        chain_tests[key] = current.test.lineno
    return findings
