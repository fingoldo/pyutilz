"""No new module-level function may go uncalled in ``src/pyutilz`` without a classification.

Every classified name lives in ``uncalled_functions_ignore.py`` with a one-line reason: deprecated aliases,
PEP 562 ``__dir__`` hooks, superseded helpers, and atexit handlers. Names that a package ``__init__`` re-exports are
public library API and are exempted by the gate itself (``public_api_from_exports``), so they are not listed. The gate
baseline holds only what remains unclassified and must only shrink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from .uncalled_functions_ignore import UNCALLED_FUNCTIONS_IGNORE

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "pyutilz"
_BASELINE_PATH = Path(__file__).resolve().parent / "_uncalled_functions_baseline.json"


def _production_files() -> list[Path]:
    return [p for p in sorted(SRC_ROOT.rglob("*.py")) if "__pycache__" not in p.parts]


def test_no_new_uncalled_functions(request):
    # py_ci_shared is imported inside the test: conftest forbids a module-level import of it.
    # It requires python>=3.9 and the requirements marker skips it on the 3.8 legs, so skip there explicitly;
    # on 3.9+ a missing install still fails the import.
    if sys.version_info < (3, 9):
        pytest.skip("py-ci-shared requires python>=3.9")
    from py_ci_shared.uncalled_functions import assert_no_new_uncalled_function

    assert_no_new_uncalled_function(
        _production_files(),
        root=REPO_ROOT,
        baseline_path=_BASELINE_PATH,
        ignore=tuple(UNCALLED_FUNCTIONS_IGNORE),
        request=request,
        public_api_from_exports=True,
    )
