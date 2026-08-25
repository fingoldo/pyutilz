"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# (path, mtime_ns, size) -> parsed tree (or None for an unparsable file).
# run_all() runs every registered scanner over the SAME file set in one
# process; each scanner independently called _safe_parse on every file
# before this cache existed, so a ~60-scanner registry re-read + re-parsed
# every file up to 60x. Keying on mtime_ns+size (not just path) means an
# in-process edit-then-rescan (e.g. a scanner's own test fixture rewriting
# a tmp file between assertions) still gets a fresh parse instead of a
# stale cached one.
_PARSE_CACHE: dict[tuple[str, int, int], "Optional[ast.Module]"] = {}

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
    # `.claude` holds Claude Code's agent worktrees, each a COMPLETE checkout of the repo being audited.
    # Without this, every scanner reports each violation once for the real tree and once more per nested
    # worktree - measured on autopsia with 49 of them, which turned 12 meta-ratchet tests red with findings
    # whose paths all read `<repo>/.claude/worktrees/agent-*/...`. A nested checkout is never the code under
    # audit, and its own tests do not run from here.
    ".claude",
})


def _iter_py_files(root: Path, exclude_dirs: frozenset[str]) -> Iterable[Path]:
    """Yield every ``.py`` file under ``root`` in a stable, sorted-by-path order, skipping
    files whose path BELOW ``root`` has a component matching ``exclude_dirs``. Where the root
    itself lives is never a reason to skip anything - see the comment on the check below.

    ``Path.rglob`` iteration order is filesystem-dependent (not guaranteed, and differs
    between platforms/filesystems in practice) -- scanners that compare files pairwise
    (e.g. duplicate_function_body, near_duplicate_function_body) pick whichever file they
    see first as the "reference" and flag the other as the duplicate. Without a stable
    order, which file gets flagged is nondeterministic across machines/CI runners, breaking
    reproducible findings and any test asserting on which file a finding names.
    """
    candidates = []
    root_resolved = root.resolve()
    for p in root.rglob("*"):
        if p.suffix not in _PY_EXTS or not p.is_file():
            continue
        # Match against the path BELOW `root` only. Matching the absolute path's parts let any ANCESTOR of
        # the scan root silence the entire scan: a checkout living under `.claude/worktrees/<agent>/` (where
        # every Claude Code agent worktree lives) excluded every file in it, so `scan_dead_public_callables`
        # returned zero findings for a whole package and every audit built on it passed vacuously. The same
        # trap is one directory away for anyone whose project sits under `build/`, `dist/`, `env/` or `venv/`.
        try:
            relative = p.resolve().relative_to(root_resolved)
        except ValueError:  # a symlink pointing outside the tree - judge it by its own path, as before
            relative = p
        if any(part in exclude_dirs for part in relative.parts):
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: p.as_posix())
    yield from candidates


def _safe_parse(path: Path) -> Optional[ast.Module]:
    """Read and ``ast.parse`` ``path`` as UTF-8, returning None on read failure (I/O, decode) or a syntax error instead of raising.

    Cached per (path, mtime_ns, size) for the life of the process -- see
    ``_PARSE_CACHE``'s module-level docstring for why this matters when
    ``run_all()`` runs dozens of scanners over the same file set.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        tree = None
    else:
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            tree = None

    _PARSE_CACHE[cache_key] = tree
    return tree


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
