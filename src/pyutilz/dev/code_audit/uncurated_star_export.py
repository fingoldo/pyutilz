"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- `from .submodule import *` with no curated __all__ anywhere ----------


def _module_defines_all(tree: ast.Module) -> bool:
    """True if ``tree``'s top level assigns ``__all__`` (``Assign`` or ``AnnAssign``)."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
            return True
    return False


def _star_import_targets(tree: ast.Module) -> list[tuple[str, int]]:
    """Return ``[(relative_module, lineno), ...]`` for every ``from .x import *`` (or
    ``from .x.y import *``) at module top level. Only relative imports are considered -- a star
    import from a third-party package is that package's own API-hygiene problem, not this
    project's."""
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level < 1:
            continue  # absolute import, not this package's own submodule
        if any(alias.name == "*" for alias in node.names):
            out.append((node.module or "", node.lineno))
    return out


def _resolve_relative_module_path(pkg_init_path: Path, dotted: str) -> Path:
    """Best-effort resolution of a relative ``from .submodule import *``'s target file, given the
    importing ``__init__.py``'s own path. Handles both a plain submodule (``submodule.py``) and a
    subpackage (``submodule/__init__.py``); returns a path that may not exist if resolution fails,
    the caller checks."""
    pkg_dir = pkg_init_path.parent
    parts = dotted.split(".") if dotted else []
    candidate_file = pkg_dir.joinpath(*parts).with_suffix(".py") if parts else pkg_dir / "__init__.py"
    if candidate_file.exists():
        return candidate_file
    candidate_pkg = pkg_dir.joinpath(*parts, "__init__.py") if parts else pkg_dir / "__init__.py"
    return candidate_pkg


def scan_uncurated_star_exports(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find an ``__init__.py`` doing ``from .submodule import *`` where NEITHER the ``__init__.py``
    itself NOR the star-imported submodule defines ``__all__`` -- every public name in that
    submodule silently becomes part of the package's public surface, including anything the
    submodule author considered private-by-convention (no leading underscore, but never meant for
    external use), and a same-named helper in two star-imported submodules silently shadows one of
    them with no signal at either import site.

    A submodule that DOES define its own ``__all__`` is not flagged even if the parent ``__init__``
    doesn't re-curate -- the submodule's author already made the public-surface decision, and
    ``import *`` respects it.

    Severity: P2 -- a real API-hygiene gap, not a crash; confirmed as a recurring pattern in the
    downstream mlframe project's audit (8 packages: core, data, estimators, inference, models,
    preprocessing, utils, training/composite/transforms -- audits/full_audit_2026-07-21/
    x_architecture_api_consistency.md F5).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if py.name != "__init__.py":
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        if _module_defines_all(tree):
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for dotted, lineno in _star_import_targets(tree):
            target_path = _resolve_relative_module_path(py, dotted)
            if not target_path.exists():
                continue  # can't resolve (e.g. a namespace package quirk) -- don't guess
            target_tree = _safe_parse(target_path)
            if target_tree is not None and _module_defines_all(target_tree):
                continue  # the submodule already curated its own public surface
            findings.append(Finding(
                check="uncurated_star_export",
                severity="P2",
                file=rel,
                line=lineno,
                snippet=_line_text(src_lines, lineno),
                detail=(
                    f"'from .{dotted} import *' with no __all__ on either this __init__.py or the "
                    f"target submodule -- every public name in {dotted!r} silently joins this "
                    f"package's public surface, and a same-named helper in a sibling star-import "
                    f"can silently shadow it with no signal at either import site."
                ),
            ))
    return findings
