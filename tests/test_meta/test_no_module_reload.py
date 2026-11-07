"""Module reload and ``sys.modules`` safety. Reload / pop / delete scanning is the central ``py_ci_shared.module_reload_safety``
gate; the bare ``sys.modules[...] = stub`` write whitelist below is pyutilz-specific and has no central equivalent.

History (2026-07-21 audit round 2): no code in this repository should call ``importlib.reload()`` without restoring the module.

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
from py_ci_shared.module_reload_safety import assert_no_unpaired_reloads

import pyutilz

_REPO_ROOT = Path(pyutilz.__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"

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
    "tests/test_dev_fixes_regression.py:191": "the RESTORE half of a try/finally -- puts the saved "
    "pyutilz.system.monitoring module back after the throwaway re-import; paired with the pop "
    "entry for the same finally block above.",
    "tests/test_image.py:99": "the restore half of the _stub_pil fixture teardown for pyutilz.core.image -- puts back "
    "the module saved on setup. The PIL modules themselves are restored by _restore_pil, which is a helper call, not "
    "a subscript write.",
    "tests/test_system_audit_fixes.py:394": "the prefect_helpers fixture's stub install; its "
    "finally block pops both the stub key and (when the fixture built it) "
    "pyutilz.system.scheduling.prefect plus the parent-package attribute.",
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


def test_no_unpaired_reload_or_sys_modules_drop_in_tests():
    """Every reload / ``sys.modules`` pop or delete in tests has a snapshot restore in its own scope (central gate)."""
    assert_no_unpaired_reloads(_TESTS_DIR)
