"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import walk_tree, _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- prose naming the caller, after the caller stopped calling ---------------------------------
#
# "Called from `_flush_rows()` on every batch boundary" is the kind of sentence a reader trusts  # audit-exempt: comment_names_missing_symbol -- prose names a caller defined outside this tree, as an example of what docstrings get wrong
# completely, because it is specific and it is checkable -- and nobody checks it. When `_flush_rows`
# is refactored to go through something else, the sentence stays, and the next person reasons about
# a call graph that has not existed for months.
#
# This is worse than a comment that has merely aged, because it points somewhere. A reader who
# doubts the behaviour goes to `_flush_rows`, finds nothing, and concludes they misread the
# docstring rather than that the docstring is wrong.
#
# Narrow by construction:
#
# * the claim has to be explicit -- "called from X" or "invoked by X" -- with X written as a
#   call: `foo()` or backticked. Prose that merely mentions a function is not a claim about the
#   call graph, and "used by" describes data far more often than it describes calls.
# * X has to exist in this tree. A name that resolves nowhere is comment_names_missing_symbol's
#   finding, not this one, and reporting it twice helps nobody.
# * the documented function has to be called by SOMETHING, or the claim is about a dead function
#   and the interesting finding is that it is dead.

# "called" or "invoked" only, and the name must be written as a CALL -- `foo()` or backticked.
# Both of the rule's first two hits came from a looser pattern:
#
# * "the client used by execute()" in `connect`'s docstring is a claim about the CLIENT, not about
#   who calls `connect`. "used by", "driven by" and "reached from" all describe data far more often
#   than they describe the call graph.
# * a `callers?:` form matched `caller: str` in an Args block, which is a parameter, not a claim.
#
# What survives is prose that can only mean one thing: `Called from `_flush_rows()``.  # audit-exempt: comment_names_missing_symbol -- prose names a caller defined outside this tree, as an example of what docstrings get wrong
_CLAIM = re.compile(r"\b(?:called|invoked)\s+(?:from|by)\s+(?:`(?P<quoted>[A-Za-z_]\w*)(?:\(\))?`|(?P<bare>[A-Za-z_]\w*)\(\))", re.I)


def _documented_callers(text: str) -> set[str]:
    """The function names this prose claims call the thing it documents."""
    return {match.group("quoted") or match.group("bare") for match in _CLAIM.finditer(text)}


def _calls_made_by(func: ast.AST) -> set[str]:
    """Every name this function calls, bare or through an attribute."""
    called: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _parse_tree_files(root: Path, exclude_dirs: frozenset[str]) -> list[tuple[str, ast.Module, list[str]]]:
    """Every parseable ``.py`` file under ``root`` as (relative path, module tree, source lines)."""
    parsed: list[tuple[str, ast.Module, list[str]]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is not None:
            parsed.append((py.relative_to(root).as_posix(), tree, py.read_text(encoding="utf-8", errors="replace").splitlines()))
    return parsed


def _call_graph(parsed: list[tuple[str, ast.Module, list[str]]]) -> tuple[dict[str, set[str]], set[str]]:
    """(function name -> names it calls, union of every name called anywhere) across the tree."""
    calls_by_function: dict[str, set[str]] = {}
    called_anywhere: set[str] = set()
    for _rel, tree, _lines in parsed:
        for node in walk_tree(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                made = _calls_made_by(node)
                calls_by_function.setdefault(node.name, set()).update(made)
                called_anywhere |= made
    return calls_by_function, called_anywhere


def _mismatched_caller_findings(
    func: ast.FunctionDef | ast.AsyncFunctionDef, rel: str, src_lines: list[str], calls_by_function: dict[str, set[str]]
) -> list[Finding]:
    """Findings for one function whose docstring names a caller that does not call it."""
    doc = ast.get_docstring(func)
    if not doc:
        return []
    findings: list[Finding] = []
    for claimed in sorted(_documented_callers(doc)):
        if claimed == func.name or claimed not in calls_by_function:
            continue
        if func.name in calls_by_function[claimed]:
            continue
        findings.append(
            Finding(
                check="docstring_names_a_caller_that_does_not_call",
                severity="P2",
                file=rel,
                line=func.lineno,
                snippet=_line_text(src_lines, func.lineno),
                detail=(
                    f"this docstring says `{func.name}` is called from `{claimed}`, and "
                    f"`{claimed}` contains no call to it. A pointer this specific is "
                    "trusted completely and checked by nobody: a reader who doubts the "
                    f"behaviour goes to `{claimed}`, finds nothing, and concludes they "
                    "misread the docstring. Name the real caller, or drop the sentence."
                ),
            )
        )
    return findings


def scan_docstring_names_a_caller_that_does_not_call(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a docstring naming a caller that does not call the function it documents.

    "Called from `_flush_rows()` on every batch boundary" is trusted completely, because it is  # audit-exempt: comment_names_missing_symbol -- prose names a caller defined outside this tree, as an example of what docstrings get wrong
    specific and checkable -- and nobody checks it. After `_flush_rows` is refactored the sentence
    stays, and the next reader reasons about a call graph that has not existed for months. It is
    worse than a comment that has merely aged: a reader who doubts the behaviour goes to
    `_flush_rows`, finds nothing, and concludes they misread the docstring.

    The named function must exist in the tree (otherwise it is comment_names_missing_symbol's
    finding) and the documented function must be called by something (otherwise the interesting
    finding is that it is dead).
    """
    findings: list[Finding] = []
    parsed = _parse_tree_files(root, exclude_dirs)
    calls_by_function, called_anywhere = _call_graph(parsed)
    for rel, tree, src_lines in parsed:
        for func in walk_tree(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if func.name not in called_anywhere:
                continue
            findings.extend(_mismatched_caller_findings(func, rel, src_lines, calls_by_function))
    return findings
