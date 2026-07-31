"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple, Union

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

_MIN_BODY_NODES = 8  # skip trivial bodies (pass, ..., a bare `return None`/`return False`/`return self._x`)


def _stripped_body(func_node: "ast.FunctionDef | ast.AsyncFunctionDef") -> list:
    """``func_node``'s body statements with a leading docstring (if any) removed."""
    body = list(getattr(func_node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _node_count(body: list) -> int:
    """Total AST node count across every statement in ``body``, recursively -- a proxy for
    real logical complexity. Counting only TOP-LEVEL statements undercounts a single
    ``try/except`` with substantial nested logic as "1 statement" (trivially filtered by a
    statement-count gate even though it's genuinely non-trivial code); counting LINES
    overcounts a one-line ``return None`` that happens to sit under a multi-line decorated
    signature. Recursive node count avoids both traps."""
    return sum(1 for stmt in body for _ in ast.walk(stmt))


def _normalized_body(body: list) -> str:
    """Structural AST dump of a (docstring-stripped) body -- not the function's
    name/signature/decorators -- with no line/col metadata, so two functions whose logic
    is byte-identical normalize to the same string regardless of name, file, formatting,
    comments, or docstring prose."""
    if not body:
        return ""
    return ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=True, include_attributes=False)


def scan_duplicate_function_body(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    min_nodes: int = _MIN_BODY_NODES,
) -> list[Finding]:
    """Find functions/methods with a STRUCTURALLY IDENTICAL body defined more than once
    across the tree (any file, any name) -- copy-pasted logic instead of one shared
    definition every call site imports.

    Confirmed in the wild during this session: a `_need_cuda()` helper (checks real CUDA
    device availability, not just that ``cupy`` is importable) got its exact body pasted
    into a dozen-plus test files instead of living once in a shared conftest.py -- each
    copy now silently drifts independently the moment one gets updated (a bug fix, a new
    edge case) and the others don't. The SAME risk applies to any helper, not just test
    fixtures: two production call sites solving the same problem identically, in two
    places, is exactly the case a later fix updates one copy and forgets the other.

    Matching is on BODY only (decorators, name, docstring, and parameter names are all
    ignored) -- two functions named differently, in different files, with different
    docstrings, but doing the exact same thing statement-for-statement, are flagged.
    Renaming a copy or wrapping it in a different docstring does not evade this check;
    only actually sharing the definition does.

    ``min_nodes`` filters out trivial matches (``pass``, ``...``, a bare ``return None``/
    ``return False``/``return self._x``) -- these are legitimate stub/interface boilerplate
    that coincidentally match ACROSS UNRELATED functions (confirmed during this scanner's
    own self-test: two IPython-notebook-detection stubs both doing nothing but
    ``return None`` matched a dozen structurally-unrelated one-line getters/setters
    elsewhere in the tree). Gating on LINE COUNT undercounts (a single-statement body can
    span several lines purely from decorator/signature formatting) and gating on TOP-LEVEL
    STATEMENT COUNT also undercounts (a single ``try/except`` with substantial nested logic
    counts as "1 statement"). Recursive AST node count avoids both traps.

    Dunder methods (``__init__``, ``__getstate__``, ``__eq__``, ...) are always skipped,
    mirroring this project's own ``ignore-magic`` docstring-coverage convention: their
    bodies routinely and LEGITIMATELY converge on the same shape by design (e.g. every
    ``__getstate__`` that drops one unpicklable attribute and returns ``self.__dict__.copy()``
    minus that key looks alike across unrelated classes) -- that's the language's own
    protocol working as intended, not copy-paste drift risk.

    Severity: P2 -- each copy still works today; the risk is silent future drift.
    """
    _FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
    by_hash: DefaultDict[str, List[Tuple[Path, _FuncNode, List[str]]]] = defaultdict(list)
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            body = _stripped_body(node)
            if _node_count(body) < min_nodes:
                continue
            key = _normalized_body(body)
            if not key:
                continue
            by_hash[key].append((py, node, src_lines))

    findings: list[Finding] = []
    for occurrences in by_hash.values():
        if len(occurrences) < 2:
            continue
        first_path, first_node, _ = occurrences[0]
        first_rel = first_path.relative_to(root).as_posix()
        for py, node, src_lines in occurrences[1:]:
            rel = py.relative_to(root).as_posix()
            findings.append(Finding(
                check="duplicate_function_body",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"def {node.name}(...) has a structurally identical body to "
                    f"{first_rel}:{first_node.lineno} (def {first_node.name}) -- extract into one "
                    "shared function/module and import it from both call sites instead of "
                    "maintaining copies that can silently drift apart."
                ),
            ))
    return findings
