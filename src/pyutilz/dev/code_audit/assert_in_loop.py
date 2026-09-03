"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse


def _is_data_sweep(node: ast.expr) -> bool:
    """Whether the loop iterates a DATA SOURCE rather than a hand-written list of cases.

    The distinction is what makes first-failure reporting matter. `for row in build():` sweeps a corpus
    whose size the reader does not know, so "one failure" and "a quarter of the corpus" look identical.
    `for x in (a, b, c):` is three hand-written cases, where the first failure IS the report.

    So: a call, excluding the builtins that wrap an iterable without being a source of one. `range` is
    excluded deliberately even though it can be huge - a `range` bound is visible at the call site, which
    is the property that matters here.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name not in {"range", "enumerate", "zip", "reversed", "sorted", "pairwise", "product", "permutations", "combinations"}


def _accumulates(body: list[ast.stmt]) -> bool:
    """Whether the loop body collects anything - an append/add/update/setdefault or a `+=`.

    A loop that already accumulates is reporting the whole set somewhere; the assert inside it is
    then usually a cheap per-item guard rather than the loop's only verdict.
    """
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.AugAssign):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"append", "add", "extend", "update", "setdefault"}:
                return True
    return False


def _enclosing_function(tree: ast.Module, lineno: int) -> str:
    """The innermost def containing ``lineno``, or ``"<module>"``.

    Part of the finding's identity, not decoration: a consumer keying a baseline on
    (check, file, detail) needs each site distinguishable, and two loops in one file very often
    iterate the same expression (``for row in edges.values()`` twice is the common shape).
    """
    best, best_start = "<module>", -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            if node.lineno <= lineno <= end and node.lineno > best_start:
                best, best_start = node.name, node.lineno
    return best


def scan_assert_in_loop_reports_only_the_first(
    root: Path | str,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """A bare ``assert`` inside a ``for`` over a real collection, so the loop stops at the FIRST bad item.

    The assertion is the loop's whole verdict, and it fires once. Every later violation is invisible -
    not merely unreported, but actively hidden behind the first, because the run ends there. On a sweep
    over real data that turns "N things are wrong" into "one thing is wrong", and the count is the part
    that tells a reader whether they are looking at a stale entry or a broken rule.

    Measured instance this was written from: a citation allow-list checked 1,642 rows with the assert
    inside the loop. It reported ONE stale author name for months. Collecting instead showed 437
    violations - a quarter of the corpus - which reframed the finding entirely: the mechanism was
    aimed the wrong way, not merely out of date.

    Reported only when the loop looks like a sweep: iterating a CALL that is a real data source (not a
    hand-written literal, not a `range`/`zip`/`enumerate` wrapper), with a body that does not already
    accumulate. The fix is always the same shape - append to a list inside
    the loop, assert on the list after it, and put the count in the message.
    """
    root = Path(root)
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For) or not _is_data_sweep(node.iter) or _accumulates(node.body):
                continue
            for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if not isinstance(stmt, ast.Assert):
                    continue
                # The iterated source and the loop variable go in `detail`, not just the line number: a
                # consumer keying a baseline on (check, file, detail) - the convention that keeps a
                # suppression from drifting onto a different site as line numbers move - would otherwise
                # silence the whole FILE for this check on the first entry.
                # `ast.dump` on the 3.8 fallback, not a fixed placeholder: a constant string makes
                # every finding in the file share one `detail`, so a single baseline entry would
                # silence the whole file -- the very outcome the paragraph above rules out.
                target = ast.unparse(node.target) if hasattr(ast, "unparse") else ast.dump(node.target)
                source = ast.unparse(node.iter) if hasattr(ast, "unparse") else ast.dump(node.iter)
                findings.append(
                    Finding(
                        check="assert_in_loop_first_failure_only",
                        severity="Low",
                        file=rel,
                        line=stmt.lineno,
                        snippet=_line_text(src_lines, stmt.lineno),
                        detail=(
                            f"{_enclosing_function(tree, stmt.lineno)}(): `for {target} in {source[:80]}` "
                            "assert inside a `for` over a collection: the run stops at the FIRST failing item, so "
                            "every later violation is hidden behind it and the COUNT - the part that says whether "
                            "this is one stale entry or a broken rule - is never reported. Collect the failures in "
                            "a list inside the loop, assert on the list after it, and name the count in the message."
                        ),
                    )
                )
                break
    return findings
