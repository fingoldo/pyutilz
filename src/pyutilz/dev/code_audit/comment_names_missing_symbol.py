"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

# --- prose that points at a symbol which does not exist ----------------------------------------
#
# A comment names a function, and the function is renamed. The comment still reads as
# authoritative, and nothing tells the reader it is now pointing at nothing.
#
# Confirmed instances in one audited codebase, all Low severity and none harmless:
#
# * A comment documenting the SQL-injection mitigation for a previous round's finding named
#   a private `_perm_err_sql` helper; the real helper is `perm_err_text_like_sql()`. That comment WAS the accepted
#   mitigation, so a broken pointer meant the mitigation could not be found.
# * A comment citing `cp.unlink()` "at line 619" when it sits at 711 -- and the finding notes this
#   is a RECURRENCE of an earlier round's identical closed finding.
#
# Restricted to backticked `name()` references, which are a deliberate citation rather than prose
# that happens to contain a word. Absolute line-number citations are reported separately, since
# they rot on every edit above them and cannot be validated at all.

# A LEADING UNDERSCORE only. That is the sole reliable "this must be local" signal, and the rule
# is worthless without it: unrestricted, `close()`, `min()`, `utcnow()` and `dropna()` gave 52 hits
# in one package with not one rotted pointer among them. Allowing any snake_case name did not help
# either -- `is_nan()`, `to_plotly_json()` and `model_dump()` are library methods too. The
# canonical case, a renamed private helper cited by its old private name, still matches.
_BACKTICKED_CALL = re.compile(r"`(_[A-Za-z0-9_]+)\(\)`")
_LINE_CITATION = re.compile(r"\b(?:at |on )?line[s]? (\d{2,5})\b", re.IGNORECASE)

# Words that look like a call but name a language builtin or a convention, not a local symbol.
_NOT_LOCAL = {"len", "int", "str", "dict", "list", "set", "print", "open", "range", "type"}


def _defined_names(root: Path, exclude_dirs: frozenset[str]) -> set[str]:
    """Every function, class, and method name defined anywhere in the tree, plus imported names."""
    names: set[str] = set()
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.update((a.asname or a.name).split(".")[-1] for a in node.names)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _comment_texts(py: Path) -> list[tuple[str, int]]:
    """(text, line) for every comment and docstring in the file."""
    out: list[tuple[str, int]] = []
    try:
        source = py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    try:
        out.extend((tok.string, tok.start[0]) for tok in tokenize.generate_tokens(io.StringIO(source).readline) if tok.type == tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    tree = _safe_parse(py)
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    out.append((doc, getattr(node, "lineno", 1)))
    return out


def scan_comment_names_missing_symbol(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find comments citing a backticked `name()` that is defined nowhere in the tree.

    Prose that points somewhere is trusted; when the pointer rots, the prose still reads as
    authoritative. In the audited codebase one such comment WAS the accepted mitigation for a
    previous round's SQL-injection finding, so a stale name meant the mitigation could not be
    located.

    Only backticked call syntax is examined -- a deliberate citation, not prose that happens to
    contain a word. Names defined ANYWHERE in the tree count as resolved, so a comment may cite a
    helper from another module without being flagged.
    """
    findings: list[Finding] = []
    known = _defined_names(root, exclude_dirs) | _NOT_LOCAL
    if not known:
        return findings

    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for text, line in _comment_texts(py):
            for name in dict.fromkeys(_BACKTICKED_CALL.findall(text)):
                if name in known:
                    continue
                findings.append(
                    Finding(
                        check="comment_names_missing_symbol",
                        severity="Low",
                        file=rel,
                        line=line,
                        snippet=_line_text(src_lines, line),
                        detail=(
                            f"prose here cites `{name}()`, which is defined nowhere in this tree. "
                            "A pointer that has rotted still reads as authoritative -- one such "
                            "comment was the accepted mitigation for an earlier finding, and named "
                            "a helper that had been renamed."
                        ),
                    )
                )
    return findings


def scan_comment_cites_absolute_line(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find comments citing an absolute line number.

    Such a citation is wrong the moment anything is inserted above its target, cannot be validated
    by any tool, and rots silently. The audited codebase found one 92 lines out of date and noted
    it was a RECURRENCE of an identical finding closed in an earlier round -- which is the argument
    for a check rather than another convention.

    The remedy is to cite the SYMBOL instead; symbol citations are validated by
    `comment_names_missing_symbol`.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        rel = py.relative_to(root).as_posix()
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for text, line in _comment_texts(py):
            match = _LINE_CITATION.search(text)
            if not match:
                continue
            findings.append(
                Finding(
                    check="comment_cites_absolute_line",
                    severity="Low",
                    file=rel,
                    line=line,
                    snippet=_line_text(src_lines, line),
                    detail=(
                        f"prose here cites absolute line {match.group(1)}. That is wrong as soon as "
                        "anything is inserted above the target and no tool can check it -- one such "
                        "citation was found 92 lines stale, itself a recurrence. Cite the symbol."
                    ),
                )
            )
    return findings
