"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import re
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


def _is_excluded(path: Path, root: Path, exclude_dirs: frozenset[str]) -> bool:
    """Whether ``path`` sits in an excluded directory BELOW ``root``.

    One implementation, called from every scanner that walks a tree, because three copies of this decision
    is how the bug it now prevents got in: two of them matched ``exclude_dirs`` against the ABSOLUTE path's
    components, so an ancestor of the scan root silenced the whole scan. Passing the same tree as a relative
    path found 359 findings and as an absolute path found 0, with no error either way.
    """
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:  # a symlink pointing outside the tree - judge it by its own path
        relative = path
    return any(part in exclude_dirs for part in relative.parts)


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
    for p in root.rglob("*"):
        if p.suffix not in _PY_EXTS or not p.is_file():
            continue
        if _is_excluded(p, root, exclude_dirs):
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

# An f-string interpolation renders as this in recovered SQL, never as nothing. Dropping it
# splices the text on either side together: `f"SELECT COUNT(*) FROM {table} WHERE ..."` collapsed
# to "... FROM  WHERE ...", and a scanner read the table name as `where`. `?` cannot start an
# identifier, so a pattern declines the query instead of misreading it.
_SQL_INTERPOLATION = "?"


def _module_sql_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level names bound to a string literal, for scanners that follow SQL held in a constant."""
    out: dict[str, str] = {}
    for stmt in tree.body:
        targets = list(stmt.targets) if isinstance(stmt, ast.Assign) else ([stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
        for target in targets:
            if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[target.id] = value.value
    return out


def _sql_text(node: ast.expr, constants: dict[str, str]) -> "Optional[str]":
    """The SQL an argument carries: a literal, an f-string, a concatenation, or a module constant.

    Shared rather than copied, because two scanners had drifting copies of it and this package's
    own duplicate_function_body check said so.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.JoinedStr):
        return "".join(value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else _SQL_INTERPOLATION for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _sql_text(node.left, constants)
        right = _sql_text(node.right, constants)
        if left is not None and right is not None:
            return left + right
    return None


def _subscript_index(node: ast.Subscript) -> ast.expr:
    """The expression inside ``obj[...]``, on every supported python.

    Up to 3.8 the parser wrapped a plain subscript in ``ast.Index``
    (``Subscript(slice=Index(value=Constant("k")))``); 3.9 removed the wrapper and stores the
    expression on ``.slice`` directly (bpo-34822). A scanner that reads ``.slice`` and expects a
    ``Constant`` therefore matched nothing at all on 3.8 -- no error, just a rule that quietly
    stopped firing. Shared so every scanner unwraps the same way instead of each rediscovering it.
    """
    index = node.slice
    if index.__class__.__name__ == "Index":  # ast.Index exists only pre-3.9
        return getattr(index, "value", index)  # type: ignore[no-any-return]
    return index

_SQL_FROM = re.compile(r"\bFROM\s+([A-Za-z_][\w.\"]*)", re.I)


def _sql_table_of(sql: str) -> "Optional[str]":
    """The table a statement reads, as written after FROM. Shared: two scanners had copies of it."""
    match = _SQL_FROM.search(sql)
    return match.group(1).strip('"').lower() if match else None
