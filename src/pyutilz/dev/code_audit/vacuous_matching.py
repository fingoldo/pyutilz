"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- vacuous truth in a matching predicate --------------------------------------------------------
#
# `all(stem in text for stem in stems)` is True when `stems` is empty, so a matcher handed nothing
# matches everything. Confirmed instance (Autopsia, 2026-07-26): a content-stem splitter on `[^a-z]`
# reduced every Cyrillic surface form from a bilingual lexicon to `[]`, so every Russian synonym
# matched every English sentence in two textbooks. 155 of 191 review rows did not contain the term
# they were filed under; the published yield had to be withdrawn from 191 rows to 42.
#
# `any(...)` is deliberately NOT flagged. On an empty sequence it returns False -- the refusing
# direction -- so it fails closed. Flagging it would roughly triple the hit count for no defect.

_GUARD_TEMPLATES = ("not {v}", "if {v}", "len({v})", "bool({v})", "{v} and ", "and {v}", "{v} or ")


def _is_guarded(name: str, function_source: str, module_source: str) -> bool:
    """Whether emptiness of ``name`` is established in this function, or in the module around it.

    The module-level fallback is not laziness: a private helper is routinely guarded by its only
    caller (`scope()` refuses on an empty stem list before it ever reaches `_covers()`), and
    flagging the helper would be a false positive on correct code.
    """
    guards = tuple(t.format(v=name) for t in _GUARD_TEMPLATES)
    return any(g in function_source for g in guards) or any(g in module_source for g in guards)


def scan_vacuous_empty_pattern_match(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``all(... for x in NAME)`` where nothing establishes that ``NAME`` is non-empty.

    Only a bare ``Name`` iterable is considered: a literal or an inline expression has no pathway by
    which a caller can silently supply an empty sequence, which is the whole mechanism of this bug.

    Severity: P1. Nothing raises, nothing logs, and the predicate reports a match on every input --
    so the failure surfaces as a suspiciously good result, which is the hardest kind to notice.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src = py.read_text(encoding="utf-8", errors="replace")
        src_lines = src.splitlines()
        rel = py.relative_to(root).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn_src = ast.get_source_segment(src, fn) or ""
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "all"):
                    continue
                if not (call.args and isinstance(call.args[0], (ast.GeneratorExp, ast.ListComp))):
                    continue
                iterable = call.args[0].generators[0].iter
                if not isinstance(iterable, ast.Name) or _is_guarded(iterable.id, fn_src, src):
                    continue
                findings.append(
                    Finding(
                        check="vacuous_empty_pattern_match",
                        severity="P1",
                        file=rel,
                        line=call.lineno,
                        snippet=_line_text(src_lines, call.lineno),
                        detail=(
                            f"all(... for ... in {iterable.id}) in {fn.name}() is True when {iterable.id} is empty, so an "
                            f"empty pattern matches every input. Write `bool({iterable.id}) and all(...)` or refuse earlier."
                        ),
                    )
                )
    return findings
