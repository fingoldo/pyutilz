"""Mutation teeth over the production lines a commit changed.

For every ``src/pyutilz`` module the diff against ``PYUTILZ_MUTATION_BASE`` (default ``HEAD~1``) touched, the shared
harness mutates only the changed lines and checks that the module's own test file notices each mutant. A surviving
mutant is a change the tests do not observe; it is either fixed by an assertion or accepted in ``mutation_survivors.json``
with a note saying why it cannot be observed. The baseline is a ratchet: a new survivor fails, an accepted one passes.

A module without a ``test_<stem>.py`` is skipped here, since the test-source parity gate already demands that file exists.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# py-ci-shared requires python>=3.9; tests/conftest.py ignores this module on the 3.8 legs, and a missing install fails here.
from py_ci_shared.baseline_ratchet import Baseline
from py_ci_shared.git_changed_lines import changed_lines, lines_for
from py_ci_shared.mutation_teeth import assert_no_new_surviving_mutant

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
BASE_REV = os.environ.get("PYUTILZ_MUTATION_BASE", "HEAD~1")
BASELINE = Baseline("mutation_survivors", directory=str(Path(__file__).resolve().parent))


def _changed_production_modules() -> list[Path]:
    """Repo-relative ``src/pyutilz`` modules that have a changed line and a ``test_<stem>.py`` next to the suite."""
    try:
        changed = changed_lines(REPO_ROOT, rev=BASE_REV, include_untracked=False)
    except (OSError, subprocess.CalledProcessError):
        return []
    modules = []
    for path in sorted(changed):
        if path.parts[:2] != ("src", "pyutilz") or path.suffix != ".py" or path.name == "__init__.py":
            continue
        if not lines_for(changed, path):
            continue
        if (TESTS_DIR / f"test_{path.stem.lstrip('_')}.py").exists():
            modules.append(path)
    return modules


_MODULES = _changed_production_modules()


@pytest.mark.skipif(os.environ.get("PYUTILZ_RUN_MUTATION") != "1", reason="mutation sweep runs only in the weekly mutation-teeth workflow (PYUTILZ_RUN_MUTATION=1)")
@pytest.mark.skipif(not _MODULES, reason=f"no changed src/pyutilz module with a dedicated test file against {BASE_REV}")
@pytest.mark.parametrize("module", _MODULES, ids=lambda p: p.as_posix())
def test_changed_lines_are_killed_by_their_module_tests(module, request):
    changed = changed_lines(REPO_ROOT, rev=BASE_REV, include_untracked=False)
    assert_no_new_surviving_mutant(
        module,
        [Path("tests") / f"test_{module.stem.lstrip('_')}.py"],
        REPO_ROOT,
        BASELINE,
        request,
        lines=lines_for(changed, module),
    )
