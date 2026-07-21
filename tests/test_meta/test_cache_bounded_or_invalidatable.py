"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: every
``@lru_cache``/``@functools.cache``-decorated function in this package must be either bounded
(a finite ``maxsize``) or, if it caches forever by design (``maxsize=None`` / bare ``@cache``,
or a zero-argument "singleton settings" shape), have some documented way to invalidate it.

Class this catches: unbounded/TTL-less process-lifetime caches. Four of seven findings in the
2026-07-21 audit's concurrency/caching report were exactly this shape -- ``llm.factory``'s
``_provider_cache`` (unbounded dict, not even ``lru_cache``, see the sibling
``test_dict_cache_has_eviction.py``), ``llm.config``'s ``get_llm_settings()`` (a bare
``@lru_cache`` on a zero-arg singleton getter -- an API-key rotation never took effect without a
process restart), ``performance.kernel_tuning.registry``'s ``_choice_cache``, and the OpenRouter
model catalogue. All four are now fixed with an explicit bound (LRU eviction, TTL, or both) --
this test is the forward guard so the NEXT one gets caught before an audit has to find it.

Two independent checks:

1. ``maxsize=None`` (or bare ``@functools.cache``/``@cache``) is ALWAYS unconditionally flagged --
   there is no legitimate reason for a truly unbounded cache in this codebase; every current
   ``@lru_cache`` use already passes an explicit finite ``maxsize``, so this starts at zero
   findings and any future occurrence is a real regression, not a style nit.
2. A ZERO-ARGUMENT function decorated with ``@lru_cache(maxsize=N)`` (the "cache the one
   settings/probe result forever" shape) must have SOME invalidation story: either a
   ``<funcname>.cache_clear()`` call somewhere in src/ or tests/ (proves a reset path exists and
   is exercised), or a freshness/TTL-shaped identifier (case-insensitive ``ttl``/``freshness``/
   ``expir``/``max_age``) anywhere in the SAME module (proves the module's caching story has some
   time-based invalidation, even if it lives in a sibling function -- this is deliberately
   coarse; per-function dataflow proof is out of scope for an AST walk). Verified against the
   current codebase: ``cache_base.py``'s ``_cpu_model_slug()`` (CPU model is invariant for the
   process, no invalidation is ever needed) and ``hw_fingerprint()`` (has both an explicit
   ``cache_clear()`` call site in tests AND a documented 7-day on-disk freshness window) both
   satisfy this -- the check starts at zero findings too.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent
_TESTS_DIR = _REPO_ROOT / "tests"

_CACHE_CLEAR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.cache_clear\(")
_FRESHNESS_RE = re.compile(r"ttl|freshness|expir|max_age", re.IGNORECASE)


def _is_lru_cache_or_cache_decorator(node: ast.expr) -> tuple[bool, bool]:
    """Return (is_cache_decorator, is_unbounded) for one decorator expression."""
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name != "lru_cache":
            return False, False
        for kw in node.keywords:
            if kw.arg == "maxsize" and isinstance(kw.value, ast.Constant) and kw.value.value is None:
                return True, True
        return True, False
    if isinstance(node, ast.Attribute):
        return node.attr == "cache", True
    if isinstance(node, ast.Name):
        return node.id == "cache", True
    return False, False


def _param_count(fn: ast.FunctionDef) -> int:
    a = fn.args
    return len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + (1 if a.vararg else 0) + (1 if a.kwarg else 0)


def _collect_lru_functions(root: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (unbounded_sites, zero_arg_sites) for every @lru_cache/@cache-decorated function.

    ``unbounded_sites`` are ``"file.py:lineno funcname (...)"`` strings for maxsize=None / bare
    @cache decorators. ``zero_arg_sites`` are ``(site, funcname)`` pairs for zero-parameter
    functions decorated with a BOUNDED @lru_cache(maxsize=N).
    """
    unbounded: list[str] = []
    zero_arg: list[tuple[str, str]] = []  # (site, funcname)
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
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                is_cache, is_unbounded = _is_lru_cache_or_cache_decorator(dec)
                if not is_cache:
                    continue
                site = f"{rel}:{node.lineno}"
                if is_unbounded:
                    unbounded.append(f"{site} {node.name} (unbounded: maxsize=None / bare @cache)")
                elif _param_count(node) == 0:
                    zero_arg.append((site, node.name))
    return unbounded, zero_arg


def _module_has_freshness_signal(py: Path) -> bool:
    try:
        src = py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(_FRESHNESS_RE.search(src))


def _cache_clear_call_exists(funcname: str) -> bool:
    for root in (_SRC_DIR, _TESTS_DIR):
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                src = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for m in _CACHE_CLEAR_RE.finditer(src):
                if m.group(1) == funcname:
                    return True
    return False


def test_no_unbounded_lru_cache():
    unbounded, _ = _collect_lru_functions(_SRC_DIR)
    if unbounded:
        pytest.fail(
            f"{len(unbounded)} unbounded cache decorator(s) found. ``maxsize=None`` / bare "
            f"``@functools.cache`` grows without limit for the life of the process -- give it an "
            f"explicit finite ``maxsize=N``, or if it's a zero-arg singleton getter, an "
            f"OrderedDict+LRU-eviction or TTL scheme like llm.config.get_llm_settings():\n  " + "\n  ".join(unbounded)
        )


def test_zero_arg_lru_cache_has_invalidation_story():
    _, zero_arg = _collect_lru_functions(_SRC_DIR)
    unresolved = []
    for site, funcname in zero_arg:
        rel_path = site.rsplit(":", 1)[0]
        py = _REPO_ROOT / rel_path
        if _cache_clear_call_exists(funcname):
            continue
        if _module_has_freshness_signal(py):
            continue
        unresolved.append(f"{site} {funcname}")
    if unresolved:
        pytest.fail(
            f"{len(unresolved)} zero-argument @lru_cache function(s) with no visible "
            f"invalidation story (no `<funcname>.cache_clear()` call anywhere in src/tests, and "
            f"no ttl/freshness/expir/max_age identifier in the same module). This is the "
            f"get_llm_settings() bug shape -- a cache that can never pick up a config/env/API-key "
            f"change without a process restart. Add a `.cache_clear()` call site (production code "
            f"or at least a test that exercises resetting it), or a documented TTL/freshness "
            f"window, or explain in a comment why this value is truly invariant for the process "
            f"lifetime (like cache_base.py's _cpu_model_slug -- CPU model never changes):\n  " + "\n  ".join(unresolved)
        )
