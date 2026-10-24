"""Tests must not assert on source text: they pass against dead code and against a comment.

Five files check a repository contract that is itself a fact about text: a version derived from the calling file's
bytes, the mypy floor written into two configs, a source lint over the package, the printed-advice registry's own
scan, and a banned comment spelling. Each is allowlisted whole-file with its reason; every other claim fails.
The baseline holds nothing and must stay empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
_BASELINE_PATH = Path(__file__).resolve().parent / "_source_text_claims_baseline.json"

_ALLOWLIST = {
    "tests/test_distributed.py": "version is the content hash of the calling file, so the test must read those bytes to pin the contract.",
    "tests/test_meta/test_mypy_gate_floor.py": "the mypy floor is written into pre-commit and CI config text; the contract is that text, read as data.",
    "tests/test_meta/test_py38_ast_compat.py": "a source lint over the package by design: it parses src/ and flags Python 3.8-unsafe AST spellings.",
    "tests/test_printed_advice_coverage_meta.py": "scans the package source for advising literals, mirroring the gate that owns the registry.",
    "tests/test_typing_lint_hygiene_audit_20260903_meta.py": "asserts a banned batch-stamped type-ignore justification is absent from src text; a lint by design.",
}


def _test_files() -> list[Path]:
    return [p for p in sorted(TESTS_ROOT.rglob("*.py")) if "__pycache__" not in p.parts]


def test_no_test_asserts_on_source_text_outside_the_allowlist(request):
    # py_ci_shared is imported inside the test: conftest forbids a module-level import of it.
    # It requires python>=3.9 and the requirements marker skips it on the 3.8 legs, so skip there explicitly;
    # on 3.9+ a missing install still fails the import.
    if sys.version_info < (3, 9):
        pytest.skip("py-ci-shared requires python>=3.9")
    from py_ci_shared.source_text_claims import assert_no_new_source_text_claims

    assert_no_new_source_text_claims(
        _test_files(),
        REPO_ROOT,
        allowlist=_ALLOWLIST,
        baseline_path=_BASELINE_PATH,
        request=request,
    )
