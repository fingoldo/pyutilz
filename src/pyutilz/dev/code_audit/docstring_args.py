"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text, _arg_names

# --- docstring Args: section missing a real parameter -----------------------
#
# Found in the wild (2026-07-21 audit, dev/benchmarking.py): a Google-style docstring's ``Args:``
# section documented only a subset of the function's actual parameters -- a caller reading the
# docstring for "what do I need to pass" has no idea a later-added parameter exists, or what it
# does. Only checked for functions that HAVE a Google-style ``Args:`` section already (writing one
# is not mandated by this scanner); once a function commits to documenting args, every real
# parameter should be listed, not a stale subset from before a signature change.

_ARGS_HEADER_RE = re.compile(r"^\s*Args:\s*$", re.MULTILINE)
_NEXT_SECTION_RE = re.compile(r"^\s*(Returns|Raises|Yields|Examples|Note|Notes|Attributes):\s*$", re.MULTILINE)
_ARG_NAME_RE = re.compile(r"^\s{4,8}(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(.*?\))?:", re.MULTILINE)

_IGNORED_PARAM_NAMES = frozenset({"self", "cls"})


def _documented_arg_names(docstring: str) -> set[str] | None:
    """Return the set of parameter names listed under a Google-style ``Args:`` section, or
    ``None`` if the docstring has no such section (nothing to check)."""
    m = _ARGS_HEADER_RE.search(docstring)
    if m is None:
        return None
    tail = docstring[m.end() :]
    end_m = _NEXT_SECTION_RE.search(tail)
    body = tail[: end_m.start()] if end_m else tail
    return {name.lstrip("*") for name in _ARG_NAME_RE.findall(body)}


def scan_docstring_args_completeness(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a function whose docstring HAS a Google-style ``Args:`` section that omits one or more
    of the function's actual parameters -- a caller reading the docstring for "what do I need to
    pass" has no idea an undocumented parameter exists.

    Only checked for functions that already commit to an ``Args:`` section (never flags a missing
    docstring/Args section entirely -- that is a separate, much noisier concern already covered by
    ``test_meta/test_public_docstrings.py``). ``self``/``cls`` are never expected to be documented.
    ``*args``/``**kwargs`` are checked by their bare name (``args``/``kwargs``), matching the
    common convention of documenting them without the star prefix.

    Severity: Low (a documentation-completeness gap, not a correctness bug -- but the exact shape
    that leaves a caller unaware a parameter exists at all).
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
            docstring = ast.get_docstring(node)
            if not docstring:
                continue
            documented = _documented_arg_names(docstring)
            if documented is None:
                continue
            real_params = [p for p in _arg_names(node) if p not in _IGNORED_PARAM_NAMES]
            missing = [p for p in real_params if p not in documented]
            if not missing:
                continue
            findings.append(Finding(
                check="docstring_args_incomplete",
                severity="Low",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"`{node.name}`'s docstring has an `Args:` section but omits parameter(s) "
                    f"{missing} -- a caller reading the docstring has no idea these exist."
                ),
            ))
    return findings
