"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- header and row built from two different places -----------------------------------------------
#
# Confirmed instance (Autopsia, 2026-07-26): a leaderboard printed its header from one f-string and
# its rows from another. One aggregated column was never emitted, so 25 header fields sat over 24 row
# fields and every column right of the gap displayed its neighbour's value -- the forbidden-endpoint
# count that drives the ranking penalty was read under the header `case`. Nothing raised; the table
# simply lied, for as long as nobody counted the columns by hand.
#
# Two shapes are decidable statically:
#
# 1. `csv.DictWriter(fieldnames=[...literal...])` written with a literal row dict whose keys differ.
#    DictWriter fills a key missing from the row with `restval` (default "") and says nothing, and
#    it is exactly the "one side gained a column" drift that produced the instance above.
# 2. Two format strings in one function, both with many replacement fields, whose field counts
#    differ -- the header/row pair before anyone paired them.
#
# The length-agreement of a *runtime* labels/cells pair is not statically decidable and belongs in a
# project test that calls both; this scanner covers the part that does not need to run.


def _literal_str_list(node: ast.AST) -> set[str] | None:
    """The string literals of a list/tuple node, or None when any element is computed."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    vals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return set(vals) if len(vals) == len(node.elts) and vals else None


def _format_field_count(node: ast.expr) -> int:
    """Number of replacement fields in an f-string (`JoinedStr`), 0 for anything else."""
    return sum(1 for v in node.values if isinstance(v, ast.FormattedValue)) if isinstance(node, ast.JoinedStr) else 0


def scan_table_header_row_drift(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    min_fields: int = 6,
) -> list[Finding]:
    """Find tabular output whose header and rows are built from two independent places.

    Reports (a) a ``csv.DictWriter`` whose literal ``fieldnames`` and literal written row disagree on
    keys, and (b) a function containing two f-strings of at least ``min_fields`` replacement fields
    each with DIFFERENT field counts, neither derived from a shared column structure -- the header
    and row of a fixed-width table that will drift the first time one side gains a column.

    ``min_fields`` exists to keep ordinary formatted log lines out: a table wide enough for this to
    matter has many columns, and a pair of two-field f-strings is not a table.

    Severity: P1 for the DictWriter key mismatch (a value is silently substituted), P2 for the
    unpaired f-strings (a latent shape, correct until edited).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            declared: set[str] | None = None
            declared_line = 0
            for call in ast.walk(fn):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "DictWriter":
                    for kw in call.keywords:
                        if kw.arg == "fieldnames":
                            literal = _literal_str_list(kw.value)
                            if literal is not None:
                                declared, declared_line = literal, call.lineno
            if declared is not None:
                for call in ast.walk(fn):
                    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "writerow" and call.args):
                        continue
                    arg = call.args[0]
                    if not isinstance(arg, ast.Dict) or any(k is None for k in arg.keys):
                        continue  # a **spread or a variable: not statically decidable, and not this shape
                    written = {k.value for k in arg.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    if written != declared:
                        findings.append(
                            Finding(
                                check="table_header_row_drift",
                                severity="P1",
                                file=rel,
                                line=declared_line,
                                snippet=_line_text(src_lines, declared_line),
                                detail=(
                                    f"csv header/row drift in {fn.name}(): header-only {sorted(declared - written)}, "
                                    f"row-only {sorted(written - declared)}. DictWriter fills a missing key with restval and says nothing."
                                ),
                            )
                        )

            wide = [(node, _format_field_count(node)) for node in ast.walk(fn) if isinstance(node, ast.JoinedStr) and _format_field_count(node) >= min_fields]
            counts = {count for _node, count in wide}
            if len(wide) >= 2 and len(counts) > 1:
                line = min(node.lineno for node, _count in wide)
                findings.append(
                    Finding(
                        check="table_header_row_drift",
                        severity="P2",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"{fn.name}() formats a wide table from independent f-strings with {sorted(counts)} fields. "
                            f"Build labels and cells from one indexed structure and zip them with strict=True."
                        ),
                    )
                )
    return findings
