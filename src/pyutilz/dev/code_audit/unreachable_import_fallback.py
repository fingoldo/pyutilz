"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- a defensive import guard for a module already imported unconditionally --------------------
#
# `try: from X import Y / except ImportError: <fallback>` where `X` is imported at module scope
# with no guard at all. The file cannot run without `X`, so the handler is unreachable -- and its
# comment usually advertises a degradation path ("minimal test environment", "optional dependency")
# that does not exist and has never run.
#
# Confirmed instance: a scraper's dead `except ImportError` whose comment claimed a tested
# minimal-environment fallback, in a file that imports the same package unconditionally fifteen
# lines above.
#
# One instance, and worth automating anyway, for a specific reason: the defensive-lazy-import idiom
# is ubiquitous, and a package whose `__init__` re-exports names through a loop makes "is this name
# importable" genuinely hard for a READER to answer while remaining a single pass for a checker.
# That asymmetry is the argument, not the count.


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``.

    Only those two plain forms. A compound condition (``if TYPE_CHECKING or X:``)
    may well execute at runtime, and treating it as type-only would hide a
    genuinely dead handler -- the thing this rule exists to find.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _package_of(rel: str) -> str:
    """The dotted PACKAGE a repo-relative file's relative imports resolve against."""
    stem = rel[:-3] if rel.endswith(".py") else rel
    parts = stem.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    else:
        parts = parts[:-1]
    return ".".join(parts)


def _resolved_module(node: ast.ImportFrom, package: str) -> "str | None":
    """The absolute dotted module an ``ImportFrom`` names, resolving ``from . import x`` too.

    Excluding relative imports outright made the rule blind in BOTH directions: a genuinely dead
    handler around `from . import util` was invisible, and a relative unconditional import never
    counted as certain.
    """
    if node.level == 0:
        return node.module
    base = package.split(".") if package else []
    prefix = base[: len(base) - node.level + 1] if len(base) >= node.level else []
    parts = [*prefix, *(node.module.split(".") if node.module else [])]
    return ".".join(parts) if parts else None


def _unconditional_module_imports(tree: ast.Module, package: str = "") -> set[str]:
    """FULL dotted module paths imported OUTSIDE any try block.

    The full path, not the top-level package. `import pyutilz.distributed` can fail on a missing
    optional dependency even where `import pyutilz` is certain, so comparing only the package
    reported every optional-submodule guard in a test suite as dead -- thirteen of this rule's
    fourteen first hits.
    """
    # DIRECT CHILDREN OF THE MODULE BODY ONLY. Walking the whole tree counted a function-local
    # lazy import, a `if sys.platform == "win32":` branch import and a TYPE_CHECKING-only import
    # as unconditional, declaring live `except ImportError` handlers dead. A statement nested in
    # ANY compound statement is conditional by construction, which subsumes the try-block and
    # TYPE_CHECKING special cases the previous version needed.
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolved_module(node, package)
            if resolved:
                names.add(resolved)
    return names


def _statements_in(body: "list[ast.stmt]") -> "Iterator[ast.stmt]":
    """Every statement in ``body``, descending into compound statements but not into nested defs."""
    todo: list[ast.stmt] = list(body)
    while todo:
        node = todo.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        todo.extend(child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt))


def scan_unreachable_import_fallback(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find `except ImportError` guards around a module the file already imports unconditionally.

    The handler cannot run, and the comment beside it typically describes a degradation path that
    has never executed -- which is worse than no comment, because a reader takes it as evidence
    that the path is supported.

    The MODULE is compared, not the individual name: a package that re-exports through its
    `__init__` can bind a name no `from X import Y` in the tree mentions, and treating that as
    unimportable would be a false positive. But the FULL dotted path is used, because
    `import pkg.optional_submodule` can fail where `import pkg` cannot.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        rel = py.relative_to(root).as_posix()
        package = _package_of(rel)
        certain = _unconditional_module_imports(tree, package)
        if not certain:
            continue
        src_lines = _read_src_lines(py)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            catches_import_error = any(
                (isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError"))
                or (isinstance(h.type, ast.Tuple) and any(isinstance(e, ast.Name) and e.id in ("ImportError", "ModuleNotFoundError") for e in h.type.elts))
                for h in node.handlers
            )
            if not catches_import_error:
                continue

            # The whole guarded block, not only its direct children: `try:` / `if True:` /
            # `import numpy` is the same guarded import one level deeper.
            for stmt in _statements_in(node.body):
                modules: set[str] = set()
                if isinstance(stmt, ast.Import):
                    modules = {a.name for a in stmt.names}
                elif isinstance(stmt, ast.ImportFrom):
                    resolved = _resolved_module(stmt, package)
                    modules = {resolved} if resolved else set()
                already = sorted(modules & certain)
                if not already:
                    continue
                findings.append(
                    Finding(
                        check="unreachable_import_fallback",
                        severity="Low",
                        file=rel,
                        line=stmt.lineno,
                        snippet=_line_text(src_lines, stmt.lineno),
                        detail=(
                            f"`{already[0]}` is imported unconditionally elsewhere in this file, so this "
                            "`except ImportError` cannot fire. The handler is dead code, and any comment "
                            "beside it describes a degradation path that has never run -- which reads as "
                            "evidence that the path is supported."
                        ),
                    )
                )
    return findings
