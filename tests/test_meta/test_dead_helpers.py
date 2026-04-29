"""PT-5 — every public helper inside ``pyutilz/llm/`` (the only
sub-package of pyutilz with internal-only helpers — providers, retry
policy, factory) must be referenced ≥ 2 times in the production corpus
(definition + ≥ 1 call site).

Other top-level pyutilz modules (``data/pandaslib.py``, ``text/strings.py``,
etc.) are PUBLIC API — users import directly from them in notebooks and
downstream code. Policing those would generate noise.

Heuristic copied from the mlframe ``test_dead_helpers.py`` ext, scope
restricted to ``llm/`` because that's the only sub-tree where
"definition + no caller" is unambiguously a bug (and not just an
exposed-but-unused-internally public symbol).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pyutilz
from pyutilz.dev.meta_test_utils import (
    consumer_corpus,
    public_top_level_symbols,
    strip_lineno,
)

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent

_IN_SCOPE_DIRS = ("llm",)

_SKIP_PATH_FRAGMENTS = ("tests", "__pycache__")
_SKIP_FILENAME_PREFIXES = ("bench_", "profile_", "_")
_SKIP_FILENAMES = {"__init__.py", "__main__.py", "version.py"}

_PUBLIC_API_WHITELIST: set[str] = {
    # ``LLMProvider`` etc. are exposed via the package-level imports;
    # tracked via the api stability snapshot, not here.
}

_USER_DEFERRED_DEAD_HELPERS: set[str] = {
    # Surfaced 2026-04-28 by this meta-test on its first run; maintainer
    # to review whether to keep, delete, or expose.
    "llm/exceptions.py::LLMTruncationError",
    "llm/openai_compat.py::parse_retry_after",
}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for py in PYUTILZ_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _SKIP_PATH_FRAGMENTS):
            continue
        rel_parts = py.relative_to(PYUTILZ_DIR).parts
        if not rel_parts or rel_parts[0] not in _IN_SCOPE_DIRS:
            continue
        if py.name in _SKIP_FILENAMES:
            continue
        if any(py.name.startswith(p) for p in _SKIP_FILENAME_PREFIXES):
            continue
        out.append(py)
    return out


def _reexport_set() -> set[str]:
    """Names re-exported from pyutilz/llm/__init__.py if any."""
    init_path = PYUTILZ_DIR / "llm" / "__init__.py"
    if not init_path.exists():
        return set()
    try:
        src = init_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", src, flags=re.DOTALL)
    if m:
        out.update(re.findall(r"['\"]([A-Za-z_]\w*)['\"]", m.group(1)))
    for m in re.finditer(r"from\s+\.\S+\s+import\s+([^\n#]+)", src):
        for name in re.split(r"[,\s]+", m.group(1).strip()):
            name = name.strip("()")
            if name and name.isidentifier() and not name.startswith("_"):
                out.add(name)
    return out


def test_no_dead_public_helpers_in_llm_subtree():
    reexports = _reexport_set()
    files = _python_files()
    if not files:
        pytest.skip("no in-scope files found")

    corpus = consumer_corpus(PYUTILZ_DIR)
    dead: list[str] = []
    total = 0
    for path in files:
        symbols = public_top_level_symbols(path)
        for name, lineno in symbols:
            total += 1
            if name in _PUBLIC_API_WHITELIST or name in reexports:
                continue
            occurrences = len(re.findall(rf"\b{re.escape(name)}\b", corpus))
            if occurrences >= 2:
                continue
            rel = path.relative_to(PYUTILZ_DIR)
            entry = f"{rel}:{lineno}::{name}"
            if strip_lineno(entry) in _USER_DEFERRED_DEAD_HELPERS:
                continue
            dead.append(entry)

    assert total > 5, f"only {total} public symbols audited — discovery broken?"
    if dead:
        pytest.fail(
            f"{len(dead)} public helper(s) in pyutilz/llm/ with no non-test "
            f"consumer (no other module references them, no __init__ "
            f"re-export). Either delete, OR re-export, OR whitelist:\n  "
            + "\n  ".join(dead[:30])
        )
