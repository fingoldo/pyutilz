"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- undeclared cross-domain imports -------------------------------------
#
# Class of bug found repeatedly in the 2026-07-21 audit (Critical: core/pythonlib.py
# unconditionally importing numba/joblib/portalocker despite README's "zero hard
# dependencies" claim; High: text/strings/__init__.py's unconditional pandas/numpy;
# text/tokenizers.py's unconditional `from pyutilz.database import db`, found
# independently by TWO different audit agents; system/parallel.py, database/db/__init__.py,
# data/polarslib.py, web/web.py, system/config.py's `import tomllib` breaking the declared
# Python 3.8/3.9/3.10 floor). All of these share one shape: a module lives under one
# pyproject.toml extras group but unconditionally imports (at module top level, not inside a
# function/try-guard) a third-party package declared only under a DIFFERENT extras group --
# silently breaking "install only what you need" for anyone who installed the group the
# module's OWN directory implies.
#
# This is deliberately pyutilz-specific (the domain->extras-group map below encodes this
# project's actual package layout), unlike most other code_audit scanners which are
# general-purpose across any Python project -- the class of bug it targets, however, is
# generalizable to any project with pyproject.toml extras groups and a domain-per-directory
# layout.

# Domain directory prefix (relative to src/pyutilz/) -> the ONE extras group that directory's
# modules may unconditionally import third-party packages from, without also being declared
# under a different group's package list. Mirrors tests/test_meta/test_optional_deps_isolation.py's
# _LEAF_MODULE_OWN_GROUP mapping, extended to whole directories.
_DOMAIN_OWN_GROUP: dict[str, str] = {
    "core/matrix": "system",  # Pillow, scipy
    "data/polarslib": "polars",
    "data/pandaslib": "pandas",
    "data/numbalib": "system",  # numba is core, but nothing extra needed -- harmless if absent from this map
    "database": "database",
    "text/strings": "nlp",
    "text/tokenizers": "nlp",
    "text/similarity": "system",  # numba is core
    "web": "web",
    "cloud": "cloud",
    "system": "system",
    "system/scheduling/prefect": "prefect",
    "llm": "llm",
    "dev/dashlib": "dash",
    "dev": "dev",
    "performance": "system",
}

# Third-party import root name -> the extras group(s) that legitimately declare it. A package
# absent from this map is either stdlib, a core pyproject.toml dependency (never flagged), or
# not yet triaged (also never flagged -- this scanner only checks packages it KNOWS the intended
# group for, to keep false positives near zero; extend this map when a new optional dependency
# is added to pyproject.toml).
_PACKAGE_OWN_GROUPS: dict[str, tuple[str, ...]] = {
    "pandas": ("pandas", "polars", "database", "nlp", "system"),
    "pyarrow": ("pandas",),
    "polars": ("polars", "pandas"),
    "dateutil": ("pandas", "polars", "database", "nlp", "system"),
    "tqdm": ("pandas", "polars", "system"),
    "pympler": ("pandas", "polars", "system"),
    "sqlalchemy": ("database",),
    "psycopg2": ("database",),
    "pymysql": ("database",),
    "redis": ("database",),
    "selenium": ("web",),
    "undetected_chromedriver": ("web",),
    "requests": ("web",),
    "grequests": ("web",),
    "fake_useragent": ("web",),
    "curl_cffi": ("web",),
    "boto3": ("cloud",),
    "spacy": ("nlp",),
    "nltk": ("nlp",),
    "jellyfish": ("nlp", "system"),
    "tiktoken": ("nlp", "llm"),
    "bs4": ("nlp",),
    "inflect": ("nlp",),
    "emoji_data_python": ("nlp",),
    "anthropic": ("llm",),
    "httpx": ("llm",),
    "tenacity": ("llm",),
    "pydantic": ("llm",),
    "pydantic_settings": ("llm",),
    "PIL": ("system",),
    "scipy": ("system",),
    "cpuinfo": ("system",),
    "GPUtil": ("system",),
    "xmltodict": ("system",),
    "cupy": ("gpu",),
    "flask": ("dash",),
    "dash": ("dash",),
    "dash_bootstrap_components": ("dash",),
    "prefect": ("prefect",),
    "tensorflow": ("tensorflow",),
}

# Stdlib module names never flagged (Python 3.10+ ships sys.stdlib_module_names).
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))


def _domain_for(rel_path: str) -> Optional[str]:
    """Return the ``_DOMAIN_OWN_GROUP`` key matching ``rel_path`` (longest-prefix match), or None."""
    best: Optional[str] = None
    for prefix in _DOMAIN_OWN_GROUP:
        if rel_path.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return best


def _top_level_import_roots(tree: ast.Module) -> list[tuple[str, int]]:
    """Return ``(root_package_name, lineno)`` for every MODULE-TOP-LEVEL ``import``/``from``
    statement (not nested inside a function/class/try/if), skipping relative imports."""
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.extend((alias.name.split(".")[0], node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import (pyutilz internal), never third-party
            if node.module:
                out.append((node.module.split(".")[0], node.lineno))
    return out


def scan_undeclared_imports(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find module-top-level third-party imports of a package declared ONLY under a pyproject.toml
    extras group different from the importing file's own domain -- "install only what you need"
    silently broken for anyone who installed the file's own domain's extra but not the other one.

    Only checks files under ``src/pyutilz`` whose domain directory is in ``_DOMAIN_OWN_GROUP`` and
    imports of packages known in ``_PACKAGE_OWN_GROUPS`` -- deliberately conservative (near-zero
    false positives) rather than exhaustive; extend both maps as pyproject.toml's extras evolve.

    Severity: P1 (undeclared cross-domain dependency -- breaks documented "install only what you
    need" for a real, non-hypothetical caller who installed exactly the file's own extras group).
    """
    findings: list[Finding] = []
    src_root = root / "src" / "pyutilz"
    if not src_root.is_dir():
        src_root = root  # allow scanning an already-src-rooted tree directly

    for py in _iter_py_files(root, exclude_dirs):
        try:
            rel_to_pkg = py.relative_to(src_root).as_posix()
        except ValueError:
            continue
        domain = _domain_for(rel_to_pkg)
        if domain is None:
            continue
        own_group = _DOMAIN_OWN_GROUP[domain]

        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for pkg_name, lineno in _top_level_import_roots(tree):
            if pkg_name in _STDLIB or pkg_name == "pyutilz":
                continue
            declared_groups = _PACKAGE_OWN_GROUPS.get(pkg_name)
            if declared_groups is None:
                continue  # not triaged -- don't guess, avoid false positives
            if own_group in declared_groups:
                continue  # legitimately declared for this file's own domain
            findings.append(Finding(
                check="undeclared_import",
                severity="P1",
                file=rel,
                line=lineno,
                snippet=_line_text(src_lines, lineno),
                detail=(
                    f"module-top-level `import {pkg_name}` -- declared under extras group(s) "
                    f"{list(declared_groups)}, not `{own_group}` (this file's own domain). "
                    f"A caller who installed only `pip install pyutilz[{own_group}]` gets "
                    f"ModuleNotFoundError importing this file."
                ),
            ))
    return findings
