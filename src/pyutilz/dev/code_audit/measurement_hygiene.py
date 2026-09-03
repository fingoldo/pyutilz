"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, is_test_file, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- numbers that claim more than the code behind them supports ------------------------------------
#
# INTEGER-ONLY PARSE OF FREE TEXT. `_count = re.search(r"\d+", text)` reads "4.10" as 4, "2,054" as 2
# and "approaching 100%" as 100, then hands the result on as a count. Confirmed in Autopsia's
# frequency ingestion (to_graph.py), where it was latent only because no decimal had reached it yet.
# The shape generalises: a bare `\d+` against prose is a truncating parse wearing a parser's clothes.
#
# THRESHOLD PINNED BELOW THE STATED RESULT. A test docstring reads "7 of 8 demonstration cards
# recover the expected cause" while the assertion is `>= 6`. The gate then cannot fail on the
# regression it was written for -- it can only fail on a second, independent one. Found twice.
#
# Both are cheap because both compare two things already present in the source.

_BARE_DIGITS_RE = re.compile(r"^\(?\\d\+\)?$")
# Digits may be GROUPED - "7,297 of 12,121", "7 297", "7_297". Without allowing the separator the leading
# \b matched only the last group, so a docstring documenting 7,297 read as documenting 297, and any
# assertion above 297 then looked like a gate set below its own documented result. Separators are stripped
# before the int(), so the claim compares as the number a reader actually sees.
_GROUPED_INT = r"\d{1,3}(?:[,_ ]\d{3})+|\d+"
_NUMBER_CLAIM_RE = re.compile(rf"\b({_GROUPED_INT})\s+(?:of|out of)\s+({_GROUPED_INT})\b|\bat least ({_GROUPED_INT})\b|\ball ({_GROUPED_INT})\b")


def _int_or_float_conversion_nearby(fn: ast.AST, lineno: int, window: int = 3) -> bool:
    """Whether the regex match is fed to int()/float() nearby - a bare pattern is not a parse."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"int", "float"} and abs(node.lineno - lineno) <= window:
            return True
    return False


def scan_regex_integer_parse(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    r"""Find a bare ``\d+`` regex over free text whose match is converted to a number.

    Only the exactly-bare pattern is flagged (``r"\d+"`` or ``r"(\d+)"``). A pattern that accounts
    for decimals, thousands separators or a unit is a real parse and passes; that distinction is the
    whole content of the rule, so widening it would destroy the signal.

    Severity: P1. It does not raise, it does not log, and it produces a plausible smaller number --
    "4.10" arrives downstream as 4 and every check on it succeeds.
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
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in {"search", "match", "findall", "fullmatch"}):
                    continue
                if not (call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                    continue
                if not _BARE_DIGITS_RE.match(call.args[0].value.replace("\\\\", "\\")):
                    continue
                if not _int_or_float_conversion_nearby(fn, call.lineno):
                    continue
                findings.append(
                    Finding(
                        check="regex_integer_parse_truncation",
                        severity="P1",
                        file=rel,
                        line=call.lineno,
                        snippet=_line_text(src_lines, call.lineno),
                        detail=(f"bare \\d+ over free text in {fn.name}() truncates silently: '4.10' -> 4, '2,054' -> 2, 'approaching 100%' -> 100. Parse decimals and separators, or refuse."),
                    )
                )
    return findings


def scan_thresholds_below_documented_result(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a test whose docstring states a result and whose assertion accepts a weaker one.

    Matches ``N of M`` / ``at least N`` / ``all N`` in a ``test_*`` function's docstring against the
    integer literals that function compares with ``>=`` or ``>``. A strictly smaller bound is
    reported: the gate has been set below the behaviour it documents, so the regression it was
    written to catch passes it.

    Severity: P2. The test still runs and can still fail -- just not on the thing its docstring says.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not is_test_file(py, root):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test_")):
                continue
            match = _NUMBER_CLAIM_RE.search(ast.get_docstring(fn) or "")
            if not match:
                continue
            claimed = int(re.sub(r"[,_ ]", "", next(g for g in match.groups() if g is not None)))
            # Only a comparison INSIDE an assert is "what the test gates on" -- an ordinary loop
            # guard such as `if i > 0:` says nothing about the documented quantity, and comparing
            # the docstring claim against every `>`/`>=` in the function reports it at the wrong
            # line. Of the real assert bounds, only the weakest (minimum) one is the effective gate.
            bounds: list[ast.Compare] = []
            for assert_node in ast.walk(fn):
                if not isinstance(assert_node, ast.Assert):
                    continue
                for cmp_node in ast.walk(assert_node.test):
                    if not (isinstance(cmp_node, ast.Compare) and len(cmp_node.ops) == 1 and isinstance(cmp_node.ops[0], (ast.Gt, ast.GtE))):
                        continue
                    right = cmp_node.comparators[0]
                    if not (isinstance(right, ast.Constant) and isinstance(right.value, int) and not isinstance(right.value, bool)):
                        continue
                    bounds.append(cmp_node)
            if not bounds:
                continue
            weakest = min(bounds, key=lambda c: c.comparators[0].value)  # type: ignore[attr-defined,union-attr]
            asserted = weakest.comparators[0].value  # type: ignore[attr-defined,union-attr]
            if asserted < claimed:
                findings.append(
                    Finding(
                        check="threshold_below_documented_result",
                        severity="P2",
                        file=rel,
                        line=weakest.lineno,
                        snippet=_line_text(src_lines, weakest.lineno),
                        detail=(f"{fn.name}() documents {claimed} but asserts >= {asserted}: the gate is set below the behaviour it describes, so a regression to {asserted} passes."),
                    )
                )
    return findings
