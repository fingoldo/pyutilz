"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

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


def _dictwriter_headers(fn: ast.AST) -> "tuple[dict[str, tuple[set[str], int]], list[tuple[set[str], int]]]":
    """``({writer variable -> (literal fieldnames, line)}, [(literal fieldnames, line), ...])``.

    Headers are kept PER WRITER VARIABLE: keeping only the last DictWriter seen in the function made
    two internally consistent writers (`w1` with ["a","b"], `w2` with ["x","y"]) report a P1 drift
    against each other. A DictWriter constructed but never bound to a name (or bound to an
    attribute) lands in the anonymous list, and can still be matched when it is the only one here.
    """
    writers: "dict[str, tuple[set[str], int]]" = {}
    anonymous: "list[tuple[set[str], int]]" = []
    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        ctor = stmt.value
        if not (isinstance(ctor.func, ast.Attribute) and ctor.func.attr == "DictWriter"):
            continue
        for kw in ctor.keywords:
            if kw.arg != "fieldnames":
                continue
            literal = _literal_str_list(kw.value)
            if literal is None:
                continue
            target = stmt.targets[0] if stmt.targets else None
            if isinstance(target, ast.Name):
                writers[target.id] = (literal, ctor.lineno)
            else:
                anonymous.append((literal, ctor.lineno))
    for call in ast.walk(fn):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "DictWriter":
            if not any(call.lineno == line for _h, line in list(writers.values()) + anonymous):
                for kw in call.keywords:
                    if kw.arg == "fieldnames":
                        literal = _literal_str_list(kw.value)
                        if literal is not None:
                            anonymous.append((literal, call.lineno))
    return writers, anonymous


def _dictwriter_drifts(fn: ast.AST) -> "list[tuple[int, set[str], set[str]]]":
    """``(header line, header-only keys, row-only keys)`` for every literal row that disagrees."""
    writers, anonymous = _dictwriter_headers(fn)
    if not writers and not anonymous:
        return []
    only = (list(writers.values()) + anonymous)[0] if len(writers) + len(anonymous) == 1 else None
    drifts: "list[tuple[int, set[str], set[str]]]" = []
    for call in ast.walk(fn):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "writerow" and call.args):
            continue
        receiver = call.func.value
        if isinstance(receiver, ast.Name) and receiver.id in writers:
            declared, declared_line = writers[receiver.id]
        elif only is not None:
            declared, declared_line = only
        else:
            continue  # cannot say which writer this row belongs to; silence beats a coin flip
        arg = call.args[0]
        if not isinstance(arg, ast.Dict) or any(k is None for k in arg.keys):
            continue  # a **spread or a variable: not statically decidable, and not this shape
        written = {k.value for k in arg.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if written != declared:
            drifts.append((declared_line, declared - written, written - declared))
    return drifts


def _unpaired_wide_fstrings(fn: ast.AST, min_fields: int) -> "tuple[int, list[int]] | None":
    """``(first line, sorted distinct field counts)`` when a function formats a wide table from
    two or more f-strings whose replacement-field counts disagree, else ``None``."""
    wide = [(node, _format_field_count(node)) for node in ast.walk(fn) if isinstance(node, ast.JoinedStr) and _format_field_count(node) >= min_fields]
    counts = {count for _node, count in wide}
    if len(wide) >= 2 and len(counts) > 1:
        return min(node.lineno for node, _count in wide), sorted(counts)
    return None


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
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for declared_line, header_only, row_only in _dictwriter_drifts(fn):
                findings.append(
                    Finding(
                        check="table_header_row_drift",
                        severity="P1",
                        file=rel,
                        line=declared_line,
                        snippet=_line_text(src_lines, declared_line),
                        detail=(
                            f"csv header/row drift in {fn.name}(): header-only {sorted(header_only)}, "
                            f"row-only {sorted(row_only)}. DictWriter fills a missing key with restval and says nothing."
                        ),
                    )
                )

            unpaired = _unpaired_wide_fstrings(fn, min_fields)
            if unpaired is not None:
                line, counts = unpaired
                findings.append(
                    Finding(
                        check="table_header_row_drift",
                        severity="P2",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"{fn.name}() formats a wide table from independent f-strings with {counts} fields. "
                            f"Build labels and cells from one indexed structure and zip them with strict=True."
                        ),
                    )
                )
    return findings
