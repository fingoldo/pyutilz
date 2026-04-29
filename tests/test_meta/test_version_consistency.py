"""E3 — meta-test that the package version is reported the same way
across every source-of-truth.

Sources checked:
  * ``pyutilz.__version__``                  (runtime attribute)
  * ``pyutilz.version.__version__``          (the version.py constant)
  * ``[project].version`` in ``pyproject.toml`` (build metadata)

Catches the failure mode where ``version.py`` is bumped but the
``pyproject.toml`` value isn't (or vice versa) — releases ship with
mismatched metadata, ``pip install pyutilz==X`` and
``pyutilz.__version__`` disagree, downstream code that pins by
``__version__`` silently breaks.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pyutilz
from pyutilz import version as version_module

REPO_ROOT = Path(pyutilz.__file__).resolve().parent.parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _read_pyproject_version() -> str | None:
    if not PYPROJECT_PATH.exists():
        return None
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    # Naive but adequate: find the first ``version = "..."`` line in the
    # ``[project]`` section. Avoids pulling in ``tomllib`` (Python 3.11+)
    # to keep this test runnable on 3.8/3.9 too.
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[project"):
            in_project = stripped.startswith("[project]")
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = False
            continue
        if not in_project:
            continue
        m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
        if m:
            return m.group(1)
    return None


def test_version_consistent_across_sources():
    sources: dict[str, str] = {}
    if hasattr(pyutilz, "__version__"):
        sources["pyutilz.__version__"] = pyutilz.__version__
    if hasattr(version_module, "__version__"):
        sources["pyutilz.version.__version__"] = version_module.__version__
    pyproject_version = _read_pyproject_version()
    if pyproject_version is not None:
        sources["pyproject.toml::[project].version"] = pyproject_version

    assert len(sources) >= 2, (
        f"version consistency test needs ≥ 2 sources to compare; got "
        f"{list(sources)}"
    )
    distinct = set(sources.values())
    if len(distinct) > 1:
        details = "\n  ".join(f"{k} = {v!r}" for k, v in sources.items())
        pytest.fail(
            f"package version disagrees across {len(distinct)} source(s):\n  "
            + details
        )
