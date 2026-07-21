"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: a function whose
docstring says it's "deprecated" must actually call ``warnings.warn(..., DeprecationWarning)`` --
a docstring-only deprecation notice is invisible to any caller who doesn't happen to read the
source (no IDE strikethrough hint from most tooling, no runtime signal a linter/CI gate could
catch, nothing shows up in a `-W error::DeprecationWarning` sweep before a removal).

Class this catches: ``pandaslib.frames.remove_stale_columns`` was documented as "Deprecated alias
for :func:`get_non_stale_columns`" but never actually warned -- fixed earlier this round (now
calls ``warnings.warn(..., DeprecationWarning)``); this is a low-noise forward guard against the
same gap recurring on a FUTURE deprecated function (verified clean, zero findings, against the
current codebase before finalizing -- every function this scan currently flags as
docstring-deprecated already warns correctly).

Mechanism: for every public top-level or method-level function, check whether the FIRST paragraph
of its docstring contains "deprecat" (case-insensitive) -- restricted to the first paragraph
specifically so a passing mention deeper in a long docstring (e.g. "unlike the now-deprecated X,
this function...") doesn't false-positive on ITSELF being deprecated. If matched, require an
``ast.Call`` to some ``*.warn(...)`` (covers both ``warnings.warn`` and re-exported/aliased
warn functions) anywhere in the function body.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent

_DEPRECATED_RE = re.compile(r"deprecat", re.IGNORECASE)


def _has_warn_call(fn: ast.AST) -> bool:
    return any(isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "warn" for sub in ast.walk(fn))


def _find_silent_deprecations(root: Path) -> list[str]:
    out: list[str] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            first_paragraph = doc.split("\n\n", 1)[0]
            if not _DEPRECATED_RE.search(first_paragraph):
                continue
            if not _has_warn_call(node):
                out.append(f"{rel}:{node.lineno} {node.name}")
    return out


def test_docstring_deprecated_functions_actually_warn():
    found = _find_silent_deprecations(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} function(s) documented as deprecated in their docstring's first "
            f"paragraph but never call warnings.warn(..., DeprecationWarning). A docstring-only "
            f"deprecation notice is invisible to any caller who doesn't read the source -- add "
            f"the warnings.warn() call (see pandaslib.frames.remove_stale_columns), or reword the "
            f"docstring if it isn't actually deprecated (e.g. only mentioning deprecation of a "
            f"DIFFERENT, sibling function):\n  " + "\n  ".join(found)
        )
