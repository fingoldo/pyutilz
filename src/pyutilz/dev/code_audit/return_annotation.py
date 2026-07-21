"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- return-annotation / actual-return-shape mismatch -----------------------
#
# CLAUDE.md's own mypy-discipline rule calls this out as a real, recurring category: "return-type
# annotations that didn't match the actual return statements (`-> float` on a function that returns
# a tuple)". mypy catches type MISMATCHES it can statically prove, but a lazy/stale annotation that
# was never updated after a refactor (declared `-> float`, one code path added later that returns a
# tuple, mypy's inference through a helper call masked it) can still slip through. This scanner is
# a narrow, syntactic double-check independent of mypy's type inference: it flags the specific
# shape of a declared concrete (non-Optional, non-``Any``, non-``None``) scalar return type
# co-occurring with a ``return`` statement whose literal shape is structurally a DIFFERENT kind
# (a tuple/list/dict/set literal, or a bare ``return``/``return None``) -- these are almost always
# either a copy-paste-stale annotation or a forgotten ``Optional[...]`` wrapper, not a legitimate
# use of duck typing.

_SCALAR_ANNOTATION_NAMES = frozenset({"int", "float", "str", "bool", "bytes"})
_CONTAINER_LITERAL_TYPES = (ast.Tuple, ast.List, ast.Dict, ast.Set)


def _annotation_name(node: ast.AST) -> str | None:
    """Return the bare name of a simple ``ast.Name`` annotation (``-> float``), else None --
    deliberately does not attempt to resolve subscripted/union annotations (``Optional[float]``,
    ``float | None``, ``Union[...]``), which are legitimate multi-shape returns by design."""
    if isinstance(node, ast.Name):
        return node.id
    return None


class _OwnReturnFinder(ast.NodeVisitor):
    """Collect ``ast.Return`` nodes belonging directly to this function, not to a nested
    function/lambda scope (whose own annotation, if any, governs those returns instead)."""

    def __init__(self) -> None:
        self.hits: list[ast.Return] = []

    def visit_Return(self, node: ast.Return) -> None:
        """Record ``node`` as belonging to the enclosing function's own scope."""
        self.hits.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Stop traversal at a nested sync function's own scope."""
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Stop traversal at a nested async function's own scope."""
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Stop traversal at a nested lambda's own scope."""
        return None


def scan_return_annotation_mismatch(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a function declared with a concrete SCALAR return annotation (``-> float``/``-> int``/
    ``-> str``/``-> bool``/``-> bytes`` -- never a subscripted/union type, which legitimately allows
    multiple shapes) that has a ``return`` statement returning a container literal
    (tuple/list/dict/set) or a bare ``return``/``return None`` -- a structural mismatch between the
    declared type and what the function actually hands back on at least one path.

    Only checks module-level and class-level ``def``s with an explicit, simple (non-subscripted)
    return annotation -- deliberately conservative to keep false positives near zero; a function
    with NO return annotation, or an ``Optional``/``Union``/generic one, is never flagged.

    Severity: P2 (a real caller-facing type contract violation, but not silent data corruption --
    the caller gets an object of the wrong shape and usually fails fast on the next attribute
    access, rather than propagating corrupted data silently).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.returns is None:
                continue
            ann_name = _annotation_name(node.returns)
            if ann_name not in _SCALAR_ANNOTATION_NAMES:
                continue
            finder = _OwnReturnFinder()
            for stmt in node.body:
                finder.visit(stmt)
            for ret in finder.hits:
                if ret.value is None:
                    findings.append(Finding(
                        check="return_annotation_mismatch",
                        severity="P2",
                        file=rel,
                        line=ret.lineno,
                        snippet=_line_text(src_lines, ret.lineno),
                        detail=(
                            f"`{node.name}` is annotated `-> {ann_name}` but has a bare `return` "
                            f"(implicitly returns None) -- either the annotation should be "
                            f"`Optional[{ann_name}]`, or this path should return a real {ann_name}."
                        ),
                    ))
                elif isinstance(ret.value, ast.Constant) and ret.value.value is None:
                    findings.append(Finding(
                        check="return_annotation_mismatch",
                        severity="P2",
                        file=rel,
                        line=ret.lineno,
                        snippet=_line_text(src_lines, ret.lineno),
                        detail=(
                            f"`{node.name}` is annotated `-> {ann_name}` but has `return None` -- "
                            f"either the annotation should be `Optional[{ann_name}]`, or this path "
                            f"should return a real {ann_name}."
                        ),
                    ))
                elif isinstance(ret.value, _CONTAINER_LITERAL_TYPES):
                    container_kind = type(ret.value).__name__.lower()
                    findings.append(Finding(
                        check="return_annotation_mismatch",
                        severity="P2",
                        file=rel,
                        line=ret.lineno,
                        snippet=_line_text(src_lines, ret.lineno),
                        detail=(
                            f"`{node.name}` is annotated `-> {ann_name}` but returns a {container_kind} "
                            f"literal here -- the declared return type doesn't match what this path "
                            f"actually hands back."
                        ),
                    ))
    return findings
