"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Iterator

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines

# --- a comment citing `file:line` that no longer resolves ------------------------------------------------

# A path with a known extension, a colon, a line number and an optional "-end" range, not preceded by a path
# character (so a URL's tail never matches).
_CITATION = re.compile(r"(?<![\w/\\.-])([\w./\\-]+\.(?:py|pyi|sql|yaml|yml|toml|md|cfg|ini|json)):(\d+)(?:-(\d+))?")


def _docstring_lines(src: str) -> set[int]:
    """First lines of the real docstrings in *src*: the leading string of the module, a class or a function.

    A triple-quoted string passed as an argument -- a test fixture writing a file that contains a citation --
    is data, not prose about this code, and reading it produced most of the false hits in a first sweep.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.add(first.lineno)
    return lines


def _comment_and_docstring_tokens(src: str) -> Iterator[tuple[int, str]]:
    """``(line, text)`` for every comment and every docstring in *src*; code, and strings in code, are never read."""
    docstrings = _docstring_lines(src)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
            elif tok.type == tokenize.STRING and tok.start[0] in docstrings:
                yield tok.start[0], tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return


def _index_tree(root: Path, exclude_dirs: frozenset[str]) -> tuple[dict[str, list[Path]], frozenset[str]]:
    """(``{basename: [paths]}`` for every file, the name of every directory), outside *exclude_dirs*.

    ``os.walk`` pruned in place, so an excluded or hidden directory (a virtualenv, ``.git``, a data dump) is
    never descended into; filtering ``rglob`` afterwards still walked all of it, 11 s on a checkout with data.
    """
    import os

    files: dict[str, list[Path]] = {}
    dirs: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs and not d.startswith("."))
        dirs.update(dirnames)
        for name in filenames:
            files.setdefault(name, []).append(Path(dirpath) / name)
    return files, frozenset(dirs)


def _resolve(ref: str, citing: Path, root: Path, index: tuple[dict[str, list[Path]], frozenset[str]]) -> tuple[Path | None, bool]:
    """(the file *ref* names, or None; whether it is decidable here at all).

    Tried in order: relative to the root, relative to the citing file's folder, then the ONE indexed file whose
    path ends with *ref*. A multi-segment reference whose leading folder names no directory anywhere in this tree
    most likely points into another repository, which this scan cannot see, so it is undecidable rather than
    missing. (Anywhere, not only at the root: ``db_saver/helpers.py`` is cited from far away from
    ``glossum/llm/db_saver``, and after that module became a package it is exactly the stale case to report.)
    """
    for candidate in (root / ref, citing.parent / ref):
        if candidate.is_file():
            return candidate, True
    files, dirs = index
    matches = [p for p in files.get(Path(ref).name, []) if p.as_posix().endswith("/" + ref) or p.name == ref]
    if len(matches) == 1:
        return matches[0], True
    if len(matches) > 1:
        return None, False  # ambiguous: several files would fit; not this scanner's call
    parts = Path(ref).parts
    if len(parts) > 1 and parts[0] not in dirs:
        return None, False
    return None, True


def scan_stale_source_citations(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a comment or docstring citing ``file:line`` where the file is gone or the line is past its end.

    Code comments point at other code by location ("mirrors" a helper at some line of a named module). The
    location rots silently when a module is split into a package or grows and shrinks, and the comment keeps
    reading as a precise pointer. Two tiers are exact and reported:

    * the cited file resolves to nothing in the tree -- renamed, split into a package, or deleted;
    * the cited line (or the end of a cited ``a-b`` range) is past the end of the file it resolves to.

    Only comments and triple-quoted strings are read, through the tokenizer, so a string literal in code is
    never mistaken for a citation. A cited path is resolved against the root, then the citing file's folder,
    then the one file in the tree whose path ends with it.

    Not reported: a path that matches several files (ambiguous), and a multi-segment path whose leading folder
    is not in this tree (it most likely cites another repository). Not attempted: whether a line that still
    exists still holds what the comment describes -- that is heuristic, and these two tiers already catch every
    reference a rename breaks.

    Relation to ``comment_cites_absolute_line``: that opt-in scanner reports EVERY absolute-line citation, true
    or not, as a style risk (225 hits in one package, most of them accurate). This one reports only the
    citations that are demonstrably wrong now, which is what lets it run by default.

    Severity: P2.
    """
    findings: list[Finding] = []
    index: tuple[dict[str, list[Path]], frozenset[str]] | None = None
    for py in _iter_py_files(root, exclude_dirs):
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # A citation in a comment is a citation in the raw text too, so a file with none skips the tokenizer, which
        # is the whole cost of this scan (8 of 9 s on a 1,500-file tree; most files cite nothing).
        if not _CITATION.search(src):
            continue
        # Read once per file rather than lazily: _read_src_lines returns an empty list on a read failure, not
        # None, so a "have I read it yet" guard could not tell the two apart.
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for lineno, text in _comment_and_docstring_tokens(src):
            for m in _CITATION.finditer(text):
                ref = m.group(1).replace("\\", "/")
                cited_line = int(m.group(3) or m.group(2))
                if index is None:
                    index = _index_tree(root, exclude_dirs)  # built only once a citation needs it
                target, decidable = _resolve(ref, py, root, index)
                if not decidable:
                    continue
                if target is None:
                    problem = f"cites `{ref}:{m.group(2)}`, and no file in the tree resolves to `{ref}`"
                else:
                    with target.open(encoding="utf-8", errors="replace") as handle:
                        length = sum(1 for _ in handle)
                    if cited_line <= length:
                        continue
                    problem = f"cites `{ref}:{cited_line}`, and `{target.relative_to(root).as_posix()}` has {length} lines"
                findings.append(Finding(
                    check="stale_source_citation",
                    severity="P2",
                    file=rel,
                    line=lineno,
                    snippet=_line_text(src_lines, lineno),
                    detail=f"{problem}. A location citation that no longer resolves reads as a precise pointer; point it at the current place, or drop the line number.",
                ))
    return findings
