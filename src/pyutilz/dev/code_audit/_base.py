"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# --- public types --------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One audit hit.

    ``check`` is the scanner name (``"mutable_default"``,
    ``"late_binding_closure"``, ``"default_via_or"``,
    ``"broad_except_swallow"``).

    ``severity`` follows the project audit convention (P0/P1/P2/Low).

    ``file`` is relative to the audit root; ``line`` is the 1-based
    line number of the offending node. ``snippet`` is the source line
    text (one line, stripped) for context. ``detail`` is a short
    free-text reason.
    """

    check: str
    severity: str
    file: str
    line: int
    snippet: str
    detail: str

    def as_md_row(self) -> str:
        """Render this finding as one Markdown table row (``| severity | check | file:line | snippet | detail |``), escaping pipe characters in the snippet and detail."""
        snip = self.snippet.replace("|", "\\|")
        detail = self.detail.replace("|", "\\|")
        return f"| {self.severity} | {self.check} | {self.file}:{self.line} | " f"`{snip}` | {detail} |"


# --- helpers -------------------------------------------------------------


_PY_EXTS = (".py",)
_DEFAULT_EXCLUDE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", ".env", ".tox",
    "build", "dist", "node_modules", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", "htmlcov",
})


def _iter_py_files(root: Path, exclude_dirs: frozenset[str]) -> Iterable[Path]:
    """Yield every ``.py`` file under ``root``, skipping files that have any path component matching ``exclude_dirs``."""
    for p in root.rglob("*"):
        if p.suffix not in _PY_EXTS or not p.is_file():
            continue
        # Skip if any parent name matches an excluded dir.
        if any(part in exclude_dirs for part in p.parts):
            continue
        yield p


def _safe_parse(path: Path) -> Optional[ast.Module]:
    """Read and ``ast.parse`` ``path`` as UTF-8, returning None on read failure (I/O, decode) or a syntax error instead of raising."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def _line_text(src_lines: list[str], lineno: int) -> str:
    """Return the stripped text of 1-based line ``lineno`` from ``src_lines``, or ``""`` if out of range."""
    if 1 <= lineno <= len(src_lines):
        return src_lines[lineno - 1].strip()
    return ""


def _arg_names(func_node: ast.AST) -> list[str]:
    """Extract every parameter name from a FunctionDef / AsyncFunctionDef."""
    args: list[str] = []
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return args
    a = func_node.args
    args.extend(arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs))
    if a.vararg:
        args.append(a.vararg.arg)
    if a.kwarg:
        args.append(a.kwarg.arg)
    return args
