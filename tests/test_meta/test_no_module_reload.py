"""Meta-test proposed after the 2026-07-21 audit round 2 meta-test review: no code in this
repository should call ``importlib.reload()`` (or ``reload`` imported directly from
``importlib``) outside an explicit, reviewed whitelist.

``importlib.reload()`` re-executes a module's top-level code IN PLACE, into the SAME
``__dict__`` any other already-imported code still references. This repo hit a real, concrete
incident from exactly this pattern: ``test_api_stability.py`` used to reload every aliased
pyutilz module to get a "clean snapshot" uncontaminated by other tests' monkeypatching. That
reload silently rebound ``pythonlib._GET_ATTR_UNSET`` (a module-level sentinel used for the
"was this parameter passed at all?" idiom) to a brand-new object. Any function already imported
by an earlier-collected test file (``from pyutilz.pythonlib import get_attr``) kept its OLD
sentinel as its own frozen parameter default (bound once at function-definition time) -- the two
objects then disagreed under ``is``, corrupting that function's behavior for the rest of the
test session, purely depending on pytest-randomly's execution order. Fixed by building a fresh,
unregistered module copy (``importlib.util.module_from_spec`` + ``exec_module``) instead of
reloading in place -- see ``test_api_stability.py``'s ``_fresh_module_copy``.

Reload has legitimate uses (re-reading an env-var-driven module constant after
``monkeypatch.setenv``), so this is a hand-curated whitelist rather than an auto-refreshed
baseline -- each entry must explain why the reloaded module has no sentinel/frozen-default
hazard for earlier-imported consumers, not just that it happened to work today.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_SRC_DIR = Path(pyutilz.__file__).resolve().parent
_TESTS_DIR = _REPO_ROOT / "tests"

# "path/to/file.py:lineno" -> reason this specific reload call is safe.
_PERMITTED_RELOAD_SITES: dict[str, str] = {
    "tests/test_llm_config.py:30": "reloads pyutilz.llm.config only, to re-read env-var-driven "
    "_SETTINGS_TTL_SECONDS after monkeypatch.setenv; config.py defines no sentinel objects "
    "(verified via grep for '= object()'); teardown_method reloads again to restore state.",
    "tests/test_llm_config.py:44": "teardown_method's restore-reload, paired with the entry above.",
    "tests/test_llm_retry.py:55": "reloads pyutilz.llm._retry only, to re-read env-var-driven "
    "PYUTILZ_LLM_MAX_RETRIES after monkeypatch.setenv; _retry.py defines no sentinel objects "
    "(verified via grep for '= object()').",
    "tests/test_meta/test_lazy_import_safety.py:136": "reloads the top-level pyutilz package "
    "__init__ only (lazy-alias plumbing) -- reload only re-executes the ONE module passed to "
    "it, so this does not cascade-reload already-imported submodules like "
    "pyutilz.core.pythonlib, and __init__.py defines no sentinel objects of its own.",
    "tests/test_pythonlib_extra2.py:494": "deliberately reloads pyutilz.core.pythonlib to prove "
    "get_attr() survives it -- pythonlib.py's only historical sentinel hazard was "
    "_GET_ATTR_UNSET, and get_attr() now captures a second `_unset` parameter from the SAME "
    "name at the SAME def-time as its own default (comparing against that, not a bare global "
    "lookup), making it provably immune to reload-driven rebinding; verified via grep that no "
    "other '= object()' sentinel exists in pythonlib.py for this reload to still endanger.",
}


def _is_reload_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "reload":
        return True
    if isinstance(func, ast.Name) and func.id == "reload":
        return True
    return False


def _find_reload_calls(root: Path) -> list[str]:
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
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_reload_call(node):
                out.append(f"{rel}:{node.lineno}")
    return out


def test_no_unreviewed_module_reload_calls():
    found = sorted(set(_find_reload_calls(_SRC_DIR)) | set(_find_reload_calls(_TESTS_DIR)))
    unreviewed = [site for site in found if site not in _PERMITTED_RELOAD_SITES]
    if unreviewed:
        pytest.fail(
            f"{len(unreviewed)} importlib.reload()/reload() call(s) with no reviewed "
            f"justification in _PERMITTED_RELOAD_SITES. importlib.reload() re-executes a "
            f"module's code IN PLACE, silently rebinding any module-level sentinel object -- "
            f"any function already imported elsewhere (from module import name) keeps its OLD "
            f"sentinel as a frozen parameter default while the module's own namespace now "
            f"resolves to the NEW one, breaking `is` identity checks for the rest of the "
            f"process. Either avoid reload (build a fresh, unregistered module copy via "
            f"importlib.util.module_from_spec + exec_module instead, see "
            f"test_api_stability.py::_fresh_module_copy), OR add a reviewed entry to "
            f"_PERMITTED_RELOAD_SITES explaining why the reloaded module has no sentinel/"
            f"frozen-default hazard for earlier-imported consumers:\n  " + "\n  ".join(unreviewed)
        )


def test_permitted_reload_sites_still_exist():
    """A stale whitelist entry (site removed/renamed without updating this dict) hides that the
    reload call is GONE -- not a problem itself, but the entry should be cleaned up so the
    whitelist stays an accurate record of what's actually reloading what."""
    found = set(_find_reload_calls(_SRC_DIR)) | set(_find_reload_calls(_TESTS_DIR))
    stale = sorted(set(_PERMITTED_RELOAD_SITES) - found)
    if stale:
        pytest.fail(f"_PERMITTED_RELOAD_SITES has entries for site(s) that no longer call reload -- clean up after the underlying edit:\n  " + "\n  ".join(stale))
