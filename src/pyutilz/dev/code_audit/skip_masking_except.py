"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- except-block that masks a real test failure as a skip ---------------


def _except_body_calls_pytest_skip(handler: ast.ExceptHandler) -> bool:
    """True if ``handler``'s body contains a call to ``pytest.skip(...)`` (module-attribute form)
    or a bare ``skip(...)`` (``from pytest import skip``)."""
    for stmt in handler.body:
        for n in ast.walk(stmt):
            if not isinstance(n, ast.Call):
                continue
            func = n.func
            if isinstance(func, ast.Attribute) and func.attr == "skip":
                if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                    return True
            if isinstance(func, ast.Name) and func.id == "skip":
                return True
    return False


def _body_is_imports_only(body: list[ast.stmt]) -> bool:
    """True if every statement in ``body`` is an Import/ImportFrom -- the legitimate optional-dep
    guard shape, where masking an ImportError as a skip is the whole point."""
    if not body:
        return False
    return all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in body)


def scan_except_skip_masks_call_under_test(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a ``try:`` block in a ``test_*.py`` file whose body does MORE than import a module
    (i.e. actually calls the function/class under test, or any other real work) and whose
    ``except`` handler calls ``pytest.skip(...)`` -- a genuine API break (a renamed kwarg, a moved
    function, a signature change) is silently reclassified as "environment doesn't support this,"
    identical to a missing optional dependency, and the test suite goes green while the regression
    it exists to catch ships unnoticed.

    The legitimate shape -- ``try: import optional_dep \\n except ImportError: pytest.skip(...)``
    -- is exempted: a try body that is ONLY Import/ImportFrom statements is never flagged, since
    masking an import failure as a skip is exactly the point of that pattern.

    Severity: P1 -- this is the same class of bug as a defensive test wrapper hiding a real
    regression, confirmed in the wild (audits/full_audit_2026-07-21/x_test_suite_architecture.md
    F1-F3 in the downstream mlframe project: three files where this exact shape silently skipped a
    genuine kwarg-contract break in a public training API instead of failing loudly).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not py.name.startswith("test_"):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if _body_is_imports_only(node.body):
                continue
            for handler in node.handlers:
                if not _except_body_calls_pytest_skip(handler):
                    continue
                findings.append(Finding(
                    check="except_skip_masks_call_under_test",
                    severity="P1",
                    file=rel,
                    line=node.lineno,
                    snippet=_line_text(src_lines, node.lineno),
                    detail=(
                        "try-block does more than import a module (calls the function/class under "
                        "test or other real work) and its except handler calls pytest.skip(...) -- "
                        "a genuine API break gets silently reclassified as a skip instead of failing "
                        "the test. If this really is an optional-dependency guard, narrow the try "
                        "body to just the import."
                    ),
                ))
    return findings
