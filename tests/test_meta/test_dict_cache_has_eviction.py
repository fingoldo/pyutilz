"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: a module-level
``dict``/``OrderedDict`` that's actually WRITTEN TO at runtime (not just declared once and read)
must show SOME evidence of bounding or cleanup, or it grows forever for the life of the process.

This is the ``_provider_cache``-shape bug class -- not just the ``@lru_cache`` variant covered by
the sibling ``test_cache_bounded_or_invalidatable.py``, but plain hand-rolled dicts used as an
insert-only cache. Writing this test surfaced two GENUINE new instances beyond the original audit
(proving the proactive-meta-test premise): ``core.safe_pickle._path_locks`` (one ``threading.Lock``
per distinct path ever ``safe_dump``/``safe_load``'d, no eviction -- fixed with reference-counted
cleanup rather than a plain LRU bound, since evicting a lock a thread is still blocked on would be
unsafe) and ``llm.token_counter._encoding_cache`` (one entry per distinct model-name string ever
resolved -- fixed with the same OrderedDict+LRU pattern as ``_provider_cache``).

Mechanism: find every module-level ``name = {}``/``dict()``/``OrderedDict()`` (or annotated
equivalent). If the same module also contains an INSERT site for that name
(``name[...] = ...``, ``name.update(``, or ``name.setdefault(``), require evidence of a bound or
cleanup path scoped to that specific name: a removal site (``name.pop(``, ``del name[``,
``name.popitem(``) OR a module-wide freshness/size-cap signal (``ttl``/``freshness``/``expir``/
``max_age``/``_MAX_SIZE``/``MAXSIZE`` -- case-insensitive, deliberately coarse since a size-cap
constant declared next to the dict, not literally inside the same function, still counts).
Flag if neither exists.

FP risk (confirmed by running this against the current codebase before finalizing): pure
registration registries (``code_audit.registry.SCANNERS``, ``kernel_tuning.registry._REGISTRY``)
match the "insert site with no removal" shape structurally, but their key space is bounded by a
small, fixed, developer-controlled set (the number of scanner/tuner modules registered at
import/discovery time), not unbounded runtime/user input -- the actual bug class this test guards
against. Hand-curated whitelist (same idiom as ``test_no_module_reload.py``'s
``_PERMITTED_RELOAD_SITES``), not an auto-refreshed baseline, since a future dict that matches this
shape should require the SAME deliberate "is this really a bounded registry?" review, not silently
inherit a pass because some earlier, unrelated dict happened to be in the baseline.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent

_FRESHNESS_RE = re.compile(r"ttl|freshness|expir|max_age|_max_size|maxsize", re.IGNORECASE)

# "path/to/file.py:name" -> reason this dict's insert-only shape is safe (bounded key space by
# construction, not unbounded runtime/user input).
_PERMITTED_UNBOUNDED_DICTS: dict[str, str] = {
    "src/pyutilz/dev/code_audit/registry.py:SCANNERS": "plugin registry populated by register() "
    "calls from a small, fixed set of scanner modules at import time -- key space is 'number of "
    "scanner files in this codebase', not runtime/user-driven input.",
    "src/pyutilz/performance/kernel_tuning/registry.py:_REGISTRY": "tuner-spec registry "
    "populated by register_tuner_spec() at module discovery time -- key space is 'number of "
    "kernel-tuning modules', not runtime/user-driven input.",
}


def _is_dict_like(value: ast.expr) -> bool:
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        return name in ("dict", "OrderedDict")
    return False


def _module_level_dict_names(tree: ast.Module) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target] if node.target else [], node.value
        if value is None or not _is_dict_like(value):
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out.append((t.id, node.lineno))
    return out


def _has_insert_site(src: str, name: str) -> bool:
    esc = re.escape(name)
    pattern = re.compile(rf"\b{esc}\s*\[[^\]]*\]\s*=(?!=)|\b{esc}\.update\(|\b{esc}\.setdefault\(")
    return bool(pattern.search(src))


def _has_removal_or_bound_evidence(src: str, name: str) -> bool:
    esc = re.escape(name)
    removal = re.compile(rf"\b{esc}\.pop\(|\bdel\s+{esc}\s*\[|\b{esc}\.popitem\(|\b{esc}\.move_to_end\(")
    if removal.search(src):
        return True
    return bool(_FRESHNESS_RE.search(src))


def _find_unbounded_dict_caches(root: Path) -> list[str]:
    out: list[str] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(_REPO_ROOT).as_posix()
        for name, lineno in _module_level_dict_names(tree):
            if not _has_insert_site(src, name):
                continue
            if _has_removal_or_bound_evidence(src, name):
                continue
            key = f"{rel}:{name}"
            if key in _PERMITTED_UNBOUNDED_DICTS:
                continue
            out.append(f"{rel}:{lineno} {name}")
    return out


def test_no_unbounded_module_level_dict_caches():
    found = _find_unbounded_dict_caches(_SRC_DIR)
    if found:
        pytest.fail(
            f"{len(found)} module-level dict/OrderedDict used as an insert-only cache with no "
            f"visible eviction/removal path. This is the _provider_cache bug shape -- one entry "
            f"per distinct key ever seen, kept forever. Either add a bound (OrderedDict + "
            f"popitem(last=False) LRU eviction, a TTL check, or reference-counted cleanup if "
            f"entries are synchronization primitives whose identity matters -- see "
            f"core.safe_pickle._path_locks), OR if the key space is genuinely small and "
            f"developer-controlled (a registration registry, not runtime/user-driven input), add "
            f"a reviewed entry to _PERMITTED_UNBOUNDED_DICTS:\n  " + "\n  ".join(found)
        )


def test_permitted_unbounded_dicts_still_exist():
    """A stale whitelist entry hides that the dict is gone or renamed -- clean up after the
    underlying edit so the whitelist stays an accurate record."""
    missing = []
    for key in _PERMITTED_UNBOUNDED_DICTS:
        rel_path, name = key.rsplit(":", 1)
        py = _REPO_ROOT / rel_path
        if not py.exists():
            missing.append(f"{key} (file no longer exists)")
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            missing.append(f"{key} (file unreadable/unparsable)")
            continue
        names = {n for n, _ in _module_level_dict_names(tree)}
        if name not in names:
            missing.append(key)
    if missing:
        pytest.fail(f"_PERMITTED_UNBOUNDED_DICTS has entries for dict(s) that no longer exist -- clean up after rename/removal:\n  " + "\n  ".join(missing))
