"""PT-8 — meta-test that ``import pyutilz`` succeeds even when every
optional dependency is masked.

pyutilz declares a few core hard dependencies (``pyproject.toml``'s
``[project.dependencies]``: numba, numpy, joblib, portalocker, psutil, plus a
Python<3.11-only tomli backport -- all either used unconditionally by
``pyutilz.core.pythonlib``/``pyutilz.system.system``, both transitively imported by
nearly every other subpackage, or (numpy) a hard transitive dependency of numba
itself); everything else lives in ``[project.optional-dependencies]``. The promise
to users: ``pip install pyutilz`` always works (core deps install automatically),
and you only get ``ImportError`` when you try to *use* a sub-feature whose extra
dep isn't installed.

This test enforces that promise: importing the top-level package and
each sub-package should NOT crash because some optional dep is missing.
A sub-package's helper functions can absolutely require an extra dep
— but the IMPORT itself must be safe.

Strategy:
  1. For each known optional-dep group (pandas, polars, database,
     web, cloud, nlp, llm), simulate the dep being missing by
     temporarily mapping its top-level packages to ``None`` in
     ``sys.modules`` (faster than uninstalling).
  2. Import the top-level pyutilz package and the relevant sub-packages
     in a fresh sub-process — must not raise.

Each scenario runs in a sub-process so the masking can't leak between
test invocations or contaminate other tests in the session.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Optional-dep groups from pyproject.toml. Each entry is the set of
# top-level package names that comprise the group's hard requirements;
# masking these in ``sys.modules`` simulates the user not having
# installed the extra.
_OPTIONAL_DEP_GROUPS: dict[str, list[str]] = {
    # numpy/psutil are NOT listed anywhere below (promoted to core pyproject.toml dependencies:
    # numba, itself core, hard-requires numpy at its own import time; pyutilz.system.system's
    # own __init__.py eagerly imports psutil via all six of its submodules) -- masking either
    # would no longer simulate a real "missing extra" scenario. Same reasoning for
    # numba/joblib/portalocker.
    #
    # pandas/tqdm/pympler were removed from every list below on 2026-09-02, for the same reason
    # numpy/psutil are absent: all three were promoted to core [project.dependencies] (2026-08-03,
    # driven by pyutilz.system.system's eager all-six-submodules import -- misc.py needs
    # tqdm+pympler and memory.py needs pandas, so ANY reach into that subpackage needs them).
    # Masking a CORE dependency does not simulate "the user skipped an extra"; it simulates a
    # broken install, and it made the prefect leaf-module scenario fail on a dependency
    # `pip install pyutilz[prefect]` genuinely does provide.
    "pandas": ["pyarrow", "polars"],
    "polars": ["polars"],
    "database": ["sqlalchemy", "psycopg2", "pymysql", "redis"],
    "web": ["selenium", "undetected_chromedriver", "requests", "grequests", "fake_useragent", "curl_cffi"],
    "cloud": ["google.cloud.storage", "boto3"],
    "nlp": ["spacy", "nltk", "jellyfish", "tiktoken", "bs4", "inflect", "emoji_data_python"],
    "llm": ["anthropic", "google.genai", "httpx", "tenacity", "pydantic", "pydantic_settings"],
    # Found 2026-07-21 audit: "system" and "gpu" were missing from this dict entirely, so
    # tqdm/pympler/Pillow/scipy/py-cpuinfo/GPUtil/xmltodict/cupy were never
    # masked/simulated-absent by ANY scenario below -- a structural blind spot that let
    # core/pythonlib.py's undeclared numba/joblib/portalocker imports ship undetected (those
    # three, plus psutil/numpy, used to live in this list before being promoted to core deps).
    "system": ["PIL", "scipy", "cpuinfo", "GPUtil", "xmltodict"],
    "gpu": ["cupy"],
    # Added 2026-09-02 audit: the same structural blind spot documented for "system"/"gpu" just
    # above still applied to the last three declared extras groups -- flask/dash/
    # dash_bootstrap_components/prefect/tensorflow were masked by NO scenario at all, so a new
    # unguarded module-level import in dev/dashlib.py or system/scheduling/prefect.py could ship
    # undeclared exactly the way the 2026-07-21 bugs did. _EXTRAS_WITHOUT_MASKING_SCENARIO below
    # now pins the dict against pyproject.toml so a newly ADDED extra cannot be forgotten either.
    # pydantic is listed here as well as under "llm": dash >= 4 depends on it directly, so a real
    # `pip install pyutilz[dash]` always brings it in and masking it would test a state that cannot occur.
    # orjson is soft-imported everywhere it is used (try/except with a stdlib fallback), so masking
    # it must leave every module importable -- that is exactly what this scenario pins.
    "speedups": ["orjson"],
    "dash": ["flask", "dash", "dash_bootstrap_components", "pydantic"],
    # requests is listed here as well as under "web": prefect depends on it, so a real
    # `pip install pyutilz[prefect]` always brings it in and masking it would test an impossible state.
    "prefect": ["prefect", "requests"],
    "tensorflow": ["tensorflow"],
}

# dateutil is deliberately masked by NO scenario: pandas is a CORE dependency, so python-dateutil
# arrives with the base install and no `pip install pyutilz[<group>]` can ever be missing it.
# Masking it modelled a state that cannot occur, and made pandas raise "Unable to import required
# dependencies: dateutil" from inside any leaf module that touches pandas. The scenarios below
# describe genuinely optional packages -- unlike spacy/nltk/etc
# below, which text/strings/webtext.py genuinely defers to call-site (`global X; if X is None:
# import Y`). _NLP_LAZY_ONLY_DEPS is the narrower subset used by
# test_tokenizers_module_imports_without_nlp_group_deps below, which specifically verifies that
# deferred-import subset doesn't leak into the plain `import pyutilz.text.tokenizers` statement
# -- masking the FULL "nlp" group there would also mask pandas, which pyutilz.text.strings
# (a tokenizers.py dependency) genuinely does need at import time, unrelated to this check.
_NLP_LAZY_ONLY_DEPS = ["spacy", "nltk", "jellyfish", "tiktoken", "bs4", "inflect", "emoji_data_python"]

# pyutilz sub-packages that should remain import-safe even when
# the corresponding optional-dep group is missing. The user can still
# touch unrelated sub-packages.
#
# pyutilz.core.pythonlib is included here (not in _LEAF_MODULE_OWN_GROUP below) because, after
# the 2026-07-21 audit fix, it has NO optional dependency at all -- numba/joblib/portalocker are
# real core pyproject.toml dependencies now, so this module must import with every optional
# group masked, the same bar as pyutilz.core/pyutilz.text's own package __init__.py files.
_ALWAYS_IMPORTABLE_SUBPACKAGES = ["pyutilz", "pyutilz.core", "pyutilz.text", "pyutilz.core.pythonlib"]

# Leaf submodules that are frequently imported directly by real callers, mapped to the ONE
# optional-dep group (from _OPTIONAL_DEP_GROUPS) each legitimately needs. Complements
# test_safe_subpackages_import_with_all_optional_deps_masked, which only probes package
# __init__.py files -- often declaration-only (just __all__, no eager submodule imports; see
# that test's own subjects) -- and never imports the actual leaf submodules where real callers'
# eager third-party imports live. Found 2026-07-21 audit: that blind spot is exactly why several
# modules shipped an undeclared unconditional third-party import undetected by this suite.
_LEAF_MODULE_OWN_GROUP: dict[str, str] = {
    "pyutilz.core.matrix": "system",  # Pillow, scipy
    "pyutilz.data.polarslib": "polars",
    "pyutilz.data.pandaslib": "pandas",
    "pyutilz.database.db": "database",
    "pyutilz.text.strings": "nlp",
    "pyutilz.text.tokenizers": "nlp",
    "pyutilz.web.web": "web",
    "pyutilz.system.parallel": "system",
    "pyutilz.system.system.memory": "system",
    # The two leaf modules behind the "dash"/"prefect" groups added above. Both are excluded from
    # the `all` extra (see pyproject.toml's comment on `all`), so nothing else in this suite ever
    # imported them; each does an unguarded module-level import of its group's packages.
    "pyutilz.dev.dashlib": "dash",
    "pyutilz.system.scheduling.prefect": "prefect",
}

# Extras groups deliberately NOT given a masking scenario in _OPTIONAL_DEP_GROUPS above, with the
# reason. Anything else declared in pyproject.toml must appear in that dict -- enforced by
# test_every_declared_extra_has_a_masking_scenario, so the "a new extra silently escapes this
# suite" failure mode cannot recur (2026-09-02 audit).
_EXTRAS_WITHOUT_MASKING_SCENARIO = {
    "dataframes": "pure alias for pyutilz[pandas,polars]; both members are covered individually",
    "all": "pure alias aggregating the domain groups, all covered individually",
    "stats": "empty documented alias -- its former sole member numpy is a core dependency now",
    "dev": "test/lint tooling, not importable-feature deps: nothing in src/pyutilz imports it",
    "docs": "mkdocs-material, a site-build tool never imported by the package",
}


def _declared_extras() -> list[str]:
    """Extras group names read straight out of pyproject.toml (the source of truth)."""
    import pathlib

    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - exercised only on the 3.8-3.10 CI legs
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():  # installed-package run, no repo checkout around
        return []
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    return sorted(data.get("project", {}).get("optional-dependencies", {}))


def test_every_declared_extra_has_a_masking_scenario():
    """Every `[project.optional-dependencies]` group must either be masked by a scenario in
    ``_OPTIONAL_DEP_GROUPS`` or be listed in ``_EXTRAS_WITHOUT_MASKING_SCENARIO`` with a reason.

    Found 2026-09-02 audit: "dash"/"prefect"/"tensorflow" had been declared as extras for months
    while this suite silently ignored them -- the identical omission that let the 2026-07-21
    undeclared-import bugs ship for "system"/"gpu". Deriving the check from pyproject.toml turns
    that recurring blind spot into a hard failure the moment a new extra is added.
    """
    declared = _declared_extras()
    if not declared:
        pytest.skip("pyproject.toml not available (running against an installed package)")
    covered = set(_OPTIONAL_DEP_GROUPS) | set(_EXTRAS_WITHOUT_MASKING_SCENARIO)
    uncovered = [e for e in declared if e not in covered]
    assert not uncovered, (
        f"extras {uncovered} are declared in pyproject.toml but no scenario in this suite ever "
        f"masks their packages -- add them to _OPTIONAL_DEP_GROUPS (plus the leaf module they "
        f"gate, in _LEAF_MODULE_OWN_GROUP), or to _EXTRAS_WITHOUT_MASKING_SCENARIO with a reason."
    )
    stale = [e for e in covered if e not in declared]
    assert not stale, f"{stale} listed in this suite but no longer declared in pyproject.toml"


def _deps_to_mask_except(own_group: str) -> list[str]:
    """Every top-level package name from every OTHER optional-dep group, EXCLUDING any package
    that also appears in ``own_group``'s own list (packages legitimately shared between two
    groups -- e.g. pandas/numpy appearing in both "pandas" and "nlp" -- must never be masked
    when testing the group that also legitimately needs them)."""
    own_deps = set(_OPTIONAL_DEP_GROUPS.get(own_group, []))
    deps_to_mask: list[str] = []
    for group, group_deps in _OPTIONAL_DEP_GROUPS.items():
        if group == own_group:
            continue
        deps_to_mask.extend(d for d in group_deps if d not in own_deps)
    return deps_to_mask


def _mask_modules_script(deps_to_mask: list[str], imports_to_try: list[str]) -> str:
    """Build the snippet run in a sub-process: blocks the listed deps,
    then imports the listed pyutilz sub-packages.

    The finder returns a real (but raising-on-exec) ModuleSpec rather than raising directly
    from find_spec() itself -- found 2026-07-21, needed to add the pyutilz.data.polarslib leaf
    module to this suite: polars' own optional-dependency machinery legitimately probes
    ``importlib.util.find_spec("boto3")`` (a pure existence check, exactly the documented
    correct pattern for a library with lazy optional integrations) without ever actually
    importing/executing the module unless that specific optional feature is used. Raising
    directly from find_spec broke that legitimate probe (indistinguishable from "genuinely
    installed" as far as find_spec's caller is concerned would be wrong either way -- but
    raising made even a non-executing existence check blow up), producing a false-positive
    "masked dep leaked" failure for a dependency pyutilz.data.polarslib never actually touches.
    Deferring the raise to exec_module() only fires it for an ACTUAL `import x` statement,
    matching a real "not pip installed" environment far more faithfully.
    """
    return (
        "import sys, importlib.abc, importlib.machinery\n"
        f"_BLOCKED = {deps_to_mask!r}\n"
        "class _BlockingLoader(importlib.abc.Loader):\n"
        "    def __init__(self, fullname):\n"
        "        self.fullname = fullname\n"
        "    def create_module(self, spec):\n"
        "        return None\n"
        "    def exec_module(self, module):\n"
        "        raise ImportError(f'(masked by PT-8 test) {self.fullname}')\n"
        "class _BlockingFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        for blocked in _BLOCKED:\n"
        "            if fullname == blocked or fullname.startswith(blocked + '.'):\n"
        "                return importlib.machinery.ModuleSpec(fullname, _BlockingLoader(fullname), is_package=True)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockingFinder())\n"
        f"for mod in {imports_to_try!r}:\n"
        "    try:\n"
        "        __import__(mod)\n"
        "    except ImportError as e:\n"
        "        msg = str(e)\n"
        "        if '(masked by PT-8 test)' in msg:\n"
        "            print(f'IMPORT_FAILED: {mod} ({msg})', file=sys.stderr)\n"
        "            sys.exit(1)\n"
        "        # Other ImportErrors are OK — likely a missing real dep we don't mask.\n"
        "        print(f'OK_IGNORE: {mod} - {msg}', file=sys.stderr)\n"
        "    except Exception as e:\n"
        "        print(f'IMPORT_RAISED: {mod} ({type(e).__name__}: {e})', file=sys.stderr)\n"
        "        sys.exit(2)\n"
        "print('OK')\n"
    )


@pytest.mark.parametrize("group_name", list(_OPTIONAL_DEP_GROUPS))
def test_top_level_pyutilz_imports_when_optional_group_missing(group_name):
    """Top-level ``import pyutilz`` MUST succeed even with every dep
    in ``group_name`` blocked. Sub-packages may not be importable —
    only the top-level promise is policed here.
    """
    deps = _OPTIONAL_DEP_GROUPS[group_name]
    script = _mask_modules_script(deps, ["pyutilz"])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import pyutilz`` failed when {group_name!r} deps "
            f"({deps}) were masked. The package is supposed to be "
            f"hard-dep-free; some sub-module is doing eager top-level "
            f"imports. stderr:\n{result.stderr}"
        )


def test_system_subpackage_imports_without_web_group_deps():
    """``pyutilz.system`` must import without selenium/undetected_chromedriver/etc installed.

    Found 2026-07-09: ``pyutilz.system.distributed`` does ``from pyutilz.web import web``,
    which triggers ``pyutilz/web/__init__.py``'s ``from . import browser, graphql, proxy, web``
    -- eagerly importing ALL FOUR web submodules, including ``browser.py``, which (before the
    fix) had an unguarded module-level ``from selenium.webdriver.common.by import By``. Any
    caller of ``pyutilz.system`` (a name with no obvious web-scraping connotation) was forced to
    have selenium installed, even though selenium lives under the separate, optional [web]
    extra. This broke mlframe's CI: it installs pyutilz's [system] extra but not [web], and
    ``mlframe.metrics._gpu_metrics`` -> ``pyutilz.performance.kernel_tuning...`` ->
    ``pyutilz.system.gpu_dispatch`` -> ``pyutilz.system`` (package init) -> ``.distributed`` ->
    ``pyutilz.web`` -> ``pyutilz.web.browser`` -> ``ModuleNotFoundError: No module named
    'selenium'``, failing ~1300 unrelated feature_selection/calibration tests at collection time.
    """
    deps = _OPTIONAL_DEP_GROUPS["web"]
    script = _mask_modules_script(deps, ["pyutilz.system"])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import pyutilz.system`` failed when the 'web' group deps ({deps}) were masked. "
            f"pyutilz.system has nothing to do with web scraping and must not require selenium "
            f"et al. to be installed. stderr:\n{result.stderr}"
        )


def test_tokenizers_module_imports_without_nlp_group_deps():
    """``pyutilz.text.tokenizers`` must import without nltk/spacy/etc installed.

    Found 2026-07-09: unlike spacy just above it in the same file (already guarded with
    ``try: import spacy ... except Exception: spacy = None``), ``nltk`` was imported at module
    level, unguarded. Broke pyutilz's own CI: the py3.9 test leg doesn't install the [nlp] extra,
    so ``tests/test_tokenizers_extra.py``'s ``from pyutilz.text.tokenizers import
    AdvancedTokenizer`` raised ``ModuleNotFoundError: No module named 'nltk'`` at collection time,
    aborting the whole test session (``Interrupted: 1 error during collection``) rather than just
    that one file's tests.
    """
    deps = _NLP_LAZY_ONLY_DEPS
    script = _mask_modules_script(deps, ["pyutilz.text.tokenizers"])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import pyutilz.text.tokenizers`` failed when spacy/nltk/jellyfish/tiktoken/"
            f"bs4/inflect/emoji_data_python ({deps}) were masked. AdvancedTokenizer may still "
            f"fail when actually USED without nltk/spacy, but the IMPORT itself must be safe. "
            f"stderr:\n{result.stderr}"
        )


@pytest.mark.parametrize("leaf_module,own_group", list(_LEAF_MODULE_OWN_GROUP.items()))
def test_leaf_submodule_imports_with_only_its_own_group_installed(leaf_module, own_group):
    """``pip install pyutilz[<own_group>]`` alone must be enough to import ``leaf_module`` --
    every OTHER optional-dep group's packages are masked (simulating a user who installed only
    the one extra matching this module's domain), while packages ``own_group`` legitimately
    shares with another group (e.g. pandas/numpy, needed by both "pandas" and "nlp") are
    correctly left unmasked. See _LEAF_MODULE_OWN_GROUP's docstring for why this test exists
    alongside test_safe_subpackages_import_with_all_optional_deps_masked below.
    """
    deps_to_mask = _deps_to_mask_except(own_group)
    script = _mask_modules_script(deps_to_mask, [leaf_module])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import {leaf_module}`` failed with every optional-dep group EXCEPT "
            f"{own_group!r} masked -- `pip install pyutilz[{own_group}]` alone is not enough "
            f"to import this module; it has an undeclared dependency on another group's "
            f"package. stderr:\n{result.stderr}"
        )


@pytest.mark.parametrize("subpkg", _ALWAYS_IMPORTABLE_SUBPACKAGES)
def test_safe_subpackages_import_with_all_optional_deps_masked(subpkg):
    """``pyutilz.core`` and friends must import cleanly with EVERYTHING
    optional masked. They shouldn't depend on anything outside stdlib.
    """
    all_deps: list[str] = []
    for group in _OPTIONAL_DEP_GROUPS.values():
        all_deps.extend(group)
    script = _mask_modules_script(all_deps, [subpkg])
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import {subpkg}`` failed with all optional deps masked. "
            f"This sub-package is meant to be stdlib-only — depending on "
            f"any optional dep at module-import time is a bug. "
            f"stderr:\n{result.stderr}"
        )
