"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import _DEFAULT_EXCLUDE_DIRS, Finding, _iter_py_files, _line_text, _safe_parse

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


def _unconditional_module_imports(tree: ast.Module) -> set[str]:
    """FULL dotted module paths imported OUTSIDE any try block.

    The full path, not the top-level package. `import pyutilz.distributed` can fail on a missing
    optional dependency even where `import pyutilz` is certain, so comparing only the package
    reported every optional-submodule guard in a test suite as dead -- thirteen of this rule's
    fourteen first hits.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        # A try block's imports are conditional by construction.
        if isinstance(node, ast.Try):
            for stmt in ast.walk(node):
                guarded.add(id(stmt))
        # ...and so are a TYPE_CHECKING block's, which never execute at runtime.
        # Without this the rule fires on the standard shape for an optional
        # dependency: a TYPE_CHECKING-only import so a function can carry a real
        # return annotation, PLUS a genuine try/except ImportError fallback. The
        # fallback there is reachable; the "unconditional" import is not an
        # import at all at runtime. Two such false positives in one downstream
        # repo (glossum synset_matcher.py, token_counter.py), both on packages
        # whose absence the code demonstrably handles.
        elif isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for stmt in ast.walk(node):
                guarded.add(id(stmt))

    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


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
        certain = _unconditional_module_imports(tree)
        if not certain:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

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

            for stmt in node.body:
                modules: set[str] = set()
                if isinstance(stmt, ast.Import):
                    modules = {a.name for a in stmt.names}
                elif isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
                    modules = {stmt.module}
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
