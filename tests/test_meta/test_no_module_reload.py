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
    "tests/test_meta/test_lazy_import_safety.py:135": "reloads the top-level pyutilz package "
    "__init__ only (lazy-alias plumbing) -- reload only re-executes the ONE module passed to "
    "it, so this does not cascade-reload already-imported submodules like "
    "pyutilz.core.pythonlib, and __init__.py defines no sentinel objects of its own.",
    "tests/test_pythonlib_extra2.py:537": "deliberately reloads pyutilz.core.pythonlib to prove "
    "get_attr() survives it -- pythonlib.py's only historical sentinel hazard was "
    "_GET_ATTR_UNSET, and get_attr() now captures a second `_unset` parameter from the SAME "
    "name at the SAME def-time as its own default (comparing against that, not a bare global "
    "lookup), making it provably immune to reload-driven rebinding; verified via grep that no "
    "other '= object()' sentinel exists in pythonlib.py for this reload to still endanger.",
}


# "path/to/file.py:lineno" -> reason this ``sys.modules.pop(<a pyutilz module>)`` is safe.
#
# Added 2026-09-02 (audit F15): ``sys.modules.pop("pkg.sub")`` followed by a re-import is the
# reload ban's blind spot. It splits module identity exactly the same way -- and worse, it does
# NOT clear the submodule binding on the PARENT package, so ``from pkg import sub`` keeps
# handing out the OLD object while ``import pkg.sub`` builds a NEW one. Any entry here must say
# how the parent-package attribute is restored, not merely that sys.modules is.
_PERMITTED_SYS_MODULES_POP_SITES: dict[str, str] = {
    "tests/test_image.py:44": "the _stub_pil fixture pops pyutilz.core.image so it re-imports "
    "against this file's MagicMock PIL stubs; teardown restores BOTH the sys.modules entry and "
    "the pyutilz.core.image PARENT-PACKAGE attribute (setattr/delattr), so no stub-built module "
    "outlives the fixture by either lookup route.",
    "tests/test_image.py:61": "the teardown half of the same fixture -- pops the stub-built "
    "module back out before reinstating the saved one; paired with the entry above.",
    "tests/test_dev_fixes_regression.py:195": "re-imports pyutilz.system.monitoring under a "
    "patched atexit.register to observe the shared executor's shutdown being registered at "
    "import time (unobservable any other way -- atexit has no introspection API). The finally "
    "block shuts down the throwaway executor and restores both sys.modules and the "
    "pyutilz.system.monitoring parent-package attribute, so tests asserting _TIMEOUT_EXECUTOR "
    "identity (tests/test_monitoring.py) are unaffected by execution order.",
    "tests/test_dev_fixes_regression.py:207": "the restore half of the same finally block.",
}

# Only pops of a pyutilz-owned module can split OUR identity; popping a third-party stub key
# (e.g. "PIL.Image") is ordinary test scaffolding and is not flagged.
_POP_PREFIXES = ("pyutilz",)


# "path/to/file.py:lineno" -> reason this bare ``sys.modules[...] = ...`` write is safe.
#
# Added 2026-09-03 (audit F11): assignment INTO sys.modules is the third route into the same
# hazard, and neither ban above covers it. A stub installed with a plain subscript write and never
# removed stays visible for the rest of the process, so a package that is not installed looks
# installed -- attribute-less -- to every later ``find_spec``/``import`` capability probe, and the
# real module built against that stub outlives it too. Unlike the pop check this one flags ANY
# module name, not just pyutilz-owned ones, because the stub-shadows-a-missing-package failure is
# precisely about third-party names. It scans tests/ only: ``src/pyutilz/__init__.py``'s lazy-alias
# writes are the production mechanism itself, not scaffolding.
#
# The safe shapes need no entry here because they are not subscript writes at all:
# ``monkeypatch.setitem(sys.modules, ...)`` (undone automatically) is the preferred form. Any entry
# below must say what removes the key again, and when.
_PERMITTED_SYS_MODULES_WRITE_SITES: dict[str, str] = {
    "tests/test_dev_fixes_regression.py:205": "the RESTORE half of a try/finally -- puts the saved "
    "pyutilz.system.monitoring module back after the throwaway re-import; paired with the pop "
    "entry for the same finally block above.",
    "tests/test_image.py:60": "the restore half of the _stub_pil fixture teardown -- reinstates the " "real PIL modules saved on setup.",
    "tests/test_image.py:63": "the restore half of the same teardown for pyutilz.core.image.",
    "tests/test_meta/test_lazy_import_safety.py:116": "installs a lazy PROXY (not a stub for a "
    "missing package) under pyutilz.<alias>, which is exactly what the production __init__ "
    "installs there anyway -- the key is occupied by an equivalent proxy before and after, so "
    "nothing later can observe a difference.",
    "tests/test_system_audit_fixes.py:343": "the prefect_helpers fixture's stub install; its "
    "finally block pops both the stub key and (when the fixture built it) "
    "pyutilz.system.scheduling.prefect plus the parent-package attribute.",
    "tests/test_web_extra.py:465": "a grequests stub inside a try/finally whose finally does " '`del sys.modules["grequests"]` before the assertions run.',
}


def _written_sys_modules_key(node: ast.Assign) -> bool:
    """True if ``node`` is a ``sys.modules[...] = ...`` subscript assignment."""
    for target in node.targets:
        if not isinstance(target, ast.Subscript):
            continue
        value = target.value
        if isinstance(value, ast.Attribute) and value.attr == "modules" and isinstance(value.value, ast.Name) and value.value.id == "sys":
            return True
    return False


def _find_sys_modules_writes(root: Path) -> list[str]:
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
            if isinstance(node, ast.Assign) and _written_sys_modules_key(node):
                out.append(f"{rel}:{node.lineno}")
    return out


def test_no_unreviewed_sys_modules_assignment():
    """``sys.modules["x"] = <stub>`` with no removal leaks a fake package into the whole session.

    The 2026-09-03 audit (F11) found a `prefect` stub installed this way for the rest of the
    process: any later in-process probe for prefect would see an installed-but-empty package, and
    which answer it got depended on file order, since test order is randomised here.
    """
    found = sorted(set(_find_sys_modules_writes(_TESTS_DIR)))
    unreviewed = [site for site in found if site not in _PERMITTED_SYS_MODULES_WRITE_SITES]
    if unreviewed:
        pytest.fail(
            f"{len(unreviewed)} bare sys.modules[...] = ... write(s) under tests/ with no reviewed "
            f"justification in _PERMITTED_SYS_MODULES_WRITE_SITES. A stub written this way and not "
            f"removed makes an uninstalled package look installed (and attribute-less) to every "
            f"later capability probe in the same process. Prefer "
            f"monkeypatch.setitem(sys.modules, ...), which is undone automatically, or a fixture "
            f"whose teardown removes the key AND anything imported against it; otherwise add a "
            f"reviewed entry saying what removes the key again:\n  " + "\n  ".join(unreviewed)
        )


def test_permitted_sys_modules_write_sites_still_exist():
    """Keep the write whitelist accurate: a stale entry means the write moved or went away."""
    found = set(_find_sys_modules_writes(_TESTS_DIR))
    stale = sorted(set(_PERMITTED_SYS_MODULES_WRITE_SITES) - found)
    if stale:
        pytest.fail("_PERMITTED_SYS_MODULES_WRITE_SITES has entries for site(s) that no longer write to sys.modules -- clean up after the underlying edit:\n  " + "\n  ".join(stale))


def _popped_pyutilz_module(node: ast.Call) -> "str | None":
    """Return the pyutilz module name in a ``sys.modules.pop(...)`` call, else None."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "pop"):
        return None
    target = func.value
    if not (isinstance(target, ast.Attribute) and target.attr == "modules" and isinstance(target.value, ast.Name) and target.value.id == "sys"):
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    name = node.args[0].value
    return name if name.split(".")[0] in _POP_PREFIXES else None


def _find_sys_modules_pops(root: Path) -> list[str]:
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
            if isinstance(node, ast.Call) and _popped_pyutilz_module(node) is not None:
                out.append(f"{rel}:{node.lineno}")
    return out


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


def test_no_unreviewed_sys_modules_pop_of_a_pyutilz_module():
    """``sys.modules.pop("pyutilz...")`` + re-import splits module identity just like reload().

    The reload ban above does not cover it, and it is strictly worse in one respect: popping the
    sys.modules entry leaves the PARENT package's attribute pointing at the old object, so the
    two import routes (``import pkg.sub`` vs ``from pkg import sub``) can hand out different
    module objects for the rest of the session -- order-dependently, since test order is
    randomised here.
    """
    found = sorted(set(_find_sys_modules_pops(_SRC_DIR)) | set(_find_sys_modules_pops(_TESTS_DIR)))
    unreviewed = [site for site in found if site not in _PERMITTED_SYS_MODULES_POP_SITES]
    if unreviewed:
        pytest.fail(
            f"{len(unreviewed)} sys.modules.pop() call(s) on a pyutilz module with no reviewed "
            f"justification in _PERMITTED_SYS_MODULES_POP_SITES. Popping the sys.modules entry "
            f"does NOT clear the submodule attribute on the parent package, so a subsequent "
            f"re-import rebinds that attribute to the newly built module and nothing restores "
            f"it -- `from pkg import sub` and `import pkg.sub` then disagree. Either avoid the "
            f"pop (monkeypatch the specific symbol, or build an unregistered module copy via "
            f"importlib.util.module_from_spec + exec_module), OR add a reviewed entry saying "
            f"how the parent-package attribute is restored:\n  " + "\n  ".join(unreviewed)
        )


def test_permitted_sys_modules_pop_sites_still_exist():
    """Keep the pop whitelist an accurate record: a stale entry means the pop moved or went away."""
    found = set(_find_sys_modules_pops(_SRC_DIR)) | set(_find_sys_modules_pops(_TESTS_DIR))
    stale = sorted(set(_PERMITTED_SYS_MODULES_POP_SITES) - found)
    if stale:
        pytest.fail("_PERMITTED_SYS_MODULES_POP_SITES has entries for site(s) that no longer pop -- clean up after the underlying edit:\n  " + "\n  ".join(stale)
)
