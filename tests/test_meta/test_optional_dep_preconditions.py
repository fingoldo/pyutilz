"""Meta-test: a test must STATE the optional dependency it needs, never assume it is installed.

Three CI rounds in a row went red for the same reason -- a test added by an audit wave exercised a
module behind an optional extra and passed on the dev box only because that extra happened to be
installed there (flask via ``pyutilz.dev.dashlib``, spacy via ``AdvancedTokenizer``). The individual
tests are fixed; this file exists so the fourth round cannot happen the same way.

Everything is derived, never hand-listed:

* ``_OPTIONAL_DEP_GROUPS`` / ``_LEAF_MODULE_OWN_GROUP`` come from ``test_optional_deps_isolation``,
  which already enumerates every extras group and the leaf modules it gates (and is itself pinned
  against ``pyproject.toml`` by ``test_every_declared_extra_has_a_masking_scenario``).
* Which of those groups CI actually installs is read out of ``.github/workflows/ci.yml``'s
  ``pip install -e ".[...]"`` line, so changing that line automatically changes what this file
  demands. Groups CI does NOT install are the ones a test may not assume.
* Requirements carrying an environment marker inside an INSTALLED group (``spacy ;
  python_version >= '3.10'``) are treated the same way: present on some legs, absent on others.
* The "X requires Y" ImportError messages raised by pyutilz's own lazy-dependency classes are
  scanned out of ``src/`` to map a class name to the package constructing it needs.

What this CANNOT catch is written out in ``test_gate_integrity``-style prose at the bottom of this
docstring, because a meta-test that overstates its reach is worse than none:

* It is static. A test that reaches an optional dependency INDIRECTLY -- through reflection, a
  fixture, a parametrised sweep over subpackages, ``hasattr`` on a lazy ``__getattr__`` -- imports
  nothing this scanner can see. That is exactly how the flask failure happened, and only running
  the suite with the dep masked would have caught it.
* It checks that a precondition is STATED, not that it is COMPLETE: a test that says
  ``importorskip("nltk")`` while also needing spacy satisfies the group-level rule. The
  class-requires check below closes that specific hole for classes whose own error message names
  the package, and nothing else.
* It only polices groups CI is currently known not to install. A dep that CI installs today is
  assumed available; the derivation means the demand appears the moment CI stops installing it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from tests.test_meta.test_optional_deps_isolation import (
    _LEAF_MODULE_OWN_GROUP,
    _OPTIONAL_DEP_GROUPS,
    _declared_extras,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"
_SRC_DIR = _REPO_ROOT / "src" / "pyutilz"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"

_CI_INSTALL_RE = re.compile(r"""pip install -e ["']?\.\[([^\]]+)\]""")
# ``raise ImportError("AdvancedTokenizer requires spacy, which ...")`` -- the idiom pyutilz uses for
# a class whose dependency is imported lazily. The package name is the first word after "requires".
_REQUIRES_RE = re.compile(r"\brequires\s+([A-Za-z_][A-Za-z0-9_.]*)")


def _ci_installed_extras() -> set:
    """Extras groups CI's install step passes to pip, read straight out of the workflow."""
    if not _CI_WORKFLOW.is_file():
        return set()
    m = _CI_INSTALL_RE.search(_CI_WORKFLOW.read_text(encoding="utf-8"))
    if not m:
        return set()
    return {part.strip() for part in m.group(1).split(",") if part.strip()}


def _marker_gated_roots() -> set:
    """Import roots of requirements that carry an environment marker inside an INSTALLED extra.

    ``spacy ; python_version >= '3.10'`` is installed on most legs and absent on 3.8/3.9, which is
    indistinguishable from "not installed at all" from a test's point of view.
    """
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - exercised only on the 3.8-3.10 CI legs
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = _REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return set()
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    known_roots = {dep for deps in _OPTIONAL_DEP_GROUPS.values() for dep in deps}
    installed = _ci_installed_extras()
    roots = set()
    for group, requirements in data.get("project", {}).get("optional-dependencies", {}).items():
        if group not in installed:
            continue  # an absent group is already covered wholesale below
        for requirement in requirements:
            if ";" not in requirement:
                continue
            name = re.split(r"[;<>=!\[ ]", requirement.strip(), maxsplit=1)[0].lower()
            for root in known_roots:
                if root.lower().replace(".", "").replace("_", "") == name.replace("-", "").replace("_", ""):
                    roots.add(root)
    return roots


def _gated_third_party_roots() -> set:
    """Top-level package names a test may NOT assume: every dep of an extras group CI does not
    install, plus marker-gated deps of the groups it does."""
    installed = _ci_installed_extras()
    roots = set()
    also_installed = set()
    for group, deps in _OPTIONAL_DEP_GROUPS.items():
        (also_installed if group in installed else roots).update(deps)
    # A dep listed by BOTH an absent and an installed group (pydantic: [dash] and [llm]) still
    # arrives with the install line, so demanding a precondition for it would be a false positive.
    roots -= also_installed
    roots.update(_marker_gated_roots())
    return roots


_PROBE_SCRIPT = """
import sys, importlib, importlib.abc, importlib.machinery
_BLOCKED = {blocked!r}


class _L(importlib.abc.Loader):
    def __init__(self, fullname):
        self.fullname = fullname

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError("(masked) " + self.fullname, name=self.fullname)


class _F(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        for b in _BLOCKED:
            if fullname == b or fullname.startswith(b + "."):
                return importlib.machinery.ModuleSpec(fullname, _L(fullname), is_package=True)
        return None


sys.meta_path.insert(0, _F())
for name in {modules!r}:
    try:
        importlib.import_module(name)
    except ImportError:
        print(name)
    except Exception:
        pass
"""


@lru_cache(maxsize=None)
def _leaf_modules_needing_a_precondition() -> frozenset:
    """Leaf modules that genuinely FAIL to import once their CI-absent group is masked.

    Membership in ``_LEAF_MODULE_OWN_GROUP`` alone is not enough to demand a precondition: several
    of those modules soft-import their group (try/except with a fallback) and import perfectly well
    without it -- ``pyutilz.web.web`` is the standing example, and CI proves it by running
    ``tests/test_web.py`` green with no ``web`` extra installed. Asking those tests for an
    importorskip would be a false positive that teaches people to add noise. So the set is measured,
    in one masked subprocess per group, rather than assumed.
    """
    installed = _ci_installed_extras()
    by_group: dict = {}
    for module, group in _LEAF_MODULE_OWN_GROUP.items():
        if group not in installed:
            by_group.setdefault(group, []).append(module)
    failing = set()
    for group, modules in sorted(by_group.items()):
        script = _PROBE_SCRIPT.format(blocked=_OPTIONAL_DEP_GROUPS[group], modules=sorted(modules))
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)  # nosec B603 - fixed argv, our own interpreter
        failing.update(line.strip() for line in proc.stdout.splitlines() if line.strip() in modules)
    return frozenset(failing)


def _gated_pyutilz_modules() -> dict:
    """``{pyutilz module path: its extras group}`` for leaf modules that both belong to a group CI
    does not install AND actually stop importing when that group is absent."""
    installed = _ci_installed_extras()
    needs = _leaf_modules_needing_a_precondition()
    return {module: group for module, group in _LEAF_MODULE_OWN_GROUP.items() if group not in installed and module in needs}


def _class_required_packages() -> dict:
    """``{ClassName: package}`` scanned out of ``src/`` for pyutilz's own lazy-dependency classes.

    Picks up ``raise ImportError("AdvancedTokenizer requires spacy, ...")`` and keys it by the class
    the raise sits in, so a test that constructs that class can be required to say so. This is the
    one check that catches an INCOMPLETE precondition rather than a missing one.
    """
    out: dict = {}
    gated = _gated_third_party_roots() | {dep for deps in _OPTIONAL_DEP_GROUPS.values() for dep in deps}
    for path in _SRC_DIR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):  # pragma: no cover - unreadable/py-version-specific source
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for node in ast.walk(cls):
                if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                    continue
                exc_name = getattr(node.exc.func, "id", None) or getattr(node.exc.func, "attr", None)
                if exc_name not in ("ImportError", "ModuleNotFoundError"):
                    continue
                for arg in node.exc.args:
                    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                        continue
                    m = _REQUIRES_RE.search(arg.value)
                    if m and m.group(1) in gated:
                        out[cls.name] = m.group(1)
    return out


def _test_files() -> list:
    return sorted(p for p in _TESTS_DIR.rglob("test_*.py") if "__pycache__" not in p.parts)


def _stated_preconditions(tree: ast.AST, source: str) -> set:
    """Package names the file states a precondition for: any ``importorskip("x")`` argument, plus
    any name appearing inside a ``skipif`` mark expression."""
    stated = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "importorskip" and node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str):
                    stated.add(value)
                    stated.add(value.split(".")[0])
    for match in re.finditer(r"skipif\((.*?)\)", source, re.DOTALL):
        stated.update(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", match.group(1)))
    return stated


def _guarded_import_lines(tree: ast.AST) -> set:
    """Line numbers of import statements sitting inside a ``try:`` block -- the other legitimate way
    to state that a dependency may be absent."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    guarded.add(child.lineno)
    return guarded


def _imported_roots(tree: ast.AST) -> list:
    """``(dotted_name, lineno)`` for every import in the file, at any nesting depth."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.module, node.lineno))
    return found


def _violations() -> list:
    gated_roots = _gated_third_party_roots()
    gated_modules = _gated_pyutilz_modules()
    class_requires = _class_required_packages()
    problems: list = []

    for path in _test_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:  # pragma: no cover - a py-version-specific test file
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        stated = _stated_preconditions(tree, source)
        guarded_lines = _guarded_import_lines(tree)

        for dotted, lineno in _imported_roots(tree):
            if lineno in guarded_lines:
                continue
            root = dotted.split(".")[0]
            if dotted in gated_modules or any(dotted.startswith(m + ".") for m in gated_modules):
                group = gated_modules.get(dotted) or next(g for m, g in gated_modules.items() if dotted.startswith(m + "."))
                if not stated & set(_OPTIONAL_DEP_GROUPS[group]):
                    problems.append(
                        f"{rel}:{lineno} imports {dotted}, gated on the '{group}' extra that CI does not install, "
                        f"with no pytest.importorskip for any of {sorted(_OPTIONAL_DEP_GROUPS[group])}"
                    )
            elif dotted in gated_roots or root in gated_roots:
                # A sibling of the same extras group counts: the group installs as a unit, so
                # ``importorskip("dash_bootstrap_components")`` already establishes that ``dash``
                # is there. Only a file that names NOTHING from the group is unguarded.
                siblings = {d for g, deps in _OPTIONAL_DEP_GROUPS.items() if (root in deps or dotted in deps) for d in deps}
                if not (({dotted, root} | siblings) & stated):
                    problems.append(f"{rel}:{lineno} imports optional dependency {dotted} with no pytest.importorskip('{root}')")

        for cls_name, package in class_requires.items():
            if not re.search(rf"\b{re.escape(cls_name)}\s*\(", source):
                continue
            if package.split(".")[0] in stated or package in stated:
                continue
            problems.append(
                f"{rel} constructs {cls_name}, which raises ImportError('... requires {package} ...') when "
                f"{package} is absent, with no pytest.importorskip('{package}')"
            )
    return sorted(problems)


def test_ci_install_line_is_parseable():
    """The whole file derives from this line; a silent parse failure would make every check vacuous."""
    if not _CI_WORKFLOW.is_file():
        pytest.skip("no .github/workflows/ci.yml (running against an installed package)")
    installed = _ci_installed_extras()
    assert installed, 'could not read the extras from ci.yml\'s `pip install -e ".[...]"` step'
    declared = set(_declared_extras())
    if declared:
        unknown = installed - declared
        assert not unknown, f"ci.yml installs extras that pyproject.toml does not declare: {sorted(unknown)}"
    assert set(_OPTIONAL_DEP_GROUPS) - installed, (
        "no extras group is left uninstalled by CI, so this file's checks would be vacuous -- if CI "
        "genuinely installs everything now, delete this file rather than let it pass silently"
    )


def test_no_test_states_a_missing_optional_dep_precondition():
    """Every test file that imports an optional-dep-gated module must say so with importorskip.

    See this module's docstring for the (real) limits: this is a static scan, so an indirect reach
    -- reflection, a fixture, a parametrised sweep -- is invisible to it.
    """
    problems = _violations()
    assert not problems, "test(s) reach an optional dependency CI may not have, without stating it:\n  " + "\n  ".join(problems)


def test_the_scanner_flags_a_planted_violation(tmp_path, monkeypatch):
    """Negative control: the scan above is only worth its runtime if it actually fires."""
    gated = sorted(_gated_third_party_roots())
    if not gated:
        pytest.skip("no CI-absent optional dependency to plant an import of")
    planted = tmp_path / "test_planted.py"
    planted.write_text(f"import {gated[0]}\n\n\ndef test_x():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(f"{__name__}._test_files", lambda: [planted])
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", tmp_path)
    problems = _violations()
    assert any(gated[0] in p for p in problems), problems
