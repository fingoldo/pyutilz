"""PT-8 — meta-test that ``import pyutilz`` succeeds even when every
optional dependency is masked.

pyutilz declares zero hard dependencies (``pyproject.toml`` says
``dependencies = []``); all real deps live in ``[project.optional-dependencies]``.
The promise to users: ``pip install pyutilz`` always works, and you only
get ``ImportError`` when you try to *use* a sub-feature whose extra dep
isn't installed.

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
    "pandas": ["pandas", "numpy", "pyarrow"],
    "polars": ["polars"],
    "database": ["sqlalchemy", "psycopg2", "pymysql"],
    "web": ["selenium", "undetected_chromedriver", "requests", "grequests",
            "fake_useragent", "curl_cffi"],
    "cloud": ["google.cloud.storage", "boto3"],
    "nlp": ["spacy", "nltk", "jellyfish", "tiktoken"],
    "llm": ["anthropic", "google.genai", "httpx", "tenacity",
            "pydantic", "pydantic_settings"],
}

# pyutilz sub-packages that should remain import-safe even when
# the corresponding optional-dep group is missing. The user can still
# touch unrelated sub-packages.
_ALWAYS_IMPORTABLE_SUBPACKAGES = ["pyutilz",
                                  "pyutilz.core",
                                  "pyutilz.text"]


def _mask_modules_script(deps_to_mask: list[str], imports_to_try: list[str]) -> str:
    """Build the snippet run in a sub-process: blocks the listed deps,
    then imports the listed pyutilz sub-packages."""
    return (
        "import sys, importlib.abc, importlib.machinery\n"
        f"_BLOCKED = {deps_to_mask!r}\n"
        "class _BlockingFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        for blocked in _BLOCKED:\n"
        "            if fullname == blocked or fullname.startswith(blocked + '.'):\n"
        "                raise ImportError(f'(masked by PT-8 test) {fullname}')\n"
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
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import pyutilz`` failed when {group_name!r} deps "
            f"({deps}) were masked. The package is supposed to be "
            f"hard-dep-free; some sub-module is doing eager top-level "
            f"imports. stderr:\n{result.stderr}"
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
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(
            f"``import {subpkg}`` failed with all optional deps masked. "
            f"This sub-package is meant to be stdlib-only — depending on "
            f"any optional dep at module-import time is a bug. "
            f"stderr:\n{result.stderr}"
        )
