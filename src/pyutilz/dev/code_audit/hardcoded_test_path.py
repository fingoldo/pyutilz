"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# A Windows drive-letter path ("D:\Machine Learning\...", "C:/Users/...") or a
# POSIX developer-home path ("/home/alice/...", "/Users/bob/...", "/root/...").
# Deliberately does NOT match a bare "/" (too many false positives on URLs,
# docstrings, regex character classes) -- only the specific shapes a real
# hardcoded personal-machine path takes.
_ABS_PATH_RE = re.compile(r"(^[A-Za-z]:[\\/]|^/home/[\w.-]+/|^/Users/[\w.-]+/|^/root/)")


def _is_test_file(path: Path) -> bool:
    """True if ``path`` looks like a test file by name or by living under a ``tests`` directory."""
    return path.name.startswith("test_") or path.name.endswith("_test.py") or "tests" in path.parts


def scan_hardcoded_absolute_path_in_test(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find a hardcoded, developer-machine-specific absolute path (a Windows
    drive letter, or a POSIX ``/home/<user>/``/``/Users/<user>/``/``/root/``
    path) as a string literal anywhere in a test file.

    Such a path exists on exactly one machine -- the author's. On every OTHER
    machine or CI runner it silently doesn't exist, so any conditional built
    on it (``if Path(...).exists(): ...`` guarding a fixture, a
    ``pytest.skip``/``importorskip`` gate, a ``try/except FileNotFoundError``
    around the literal) permanently no-ops there. The test still collects,
    still reports green (skipped, not failed), and never shows up in a grep
    for ``@pytest.mark.skip`` -- it occupies the mental slot of "covered"
    while providing zero regression coverage on any machine but the one that
    wrote it.

    Deliberately narrow: only flags the specific hardcoded-personal-path
    SHAPE (drive letter, or a home-directory-rooted path) -- a bare ``/tmp/``
    or ``/var/``-rooted literal is common, portable, and NOT flagged, and a
    path built from ``Path(__file__).parent`` / ``tmp_path`` / an env var is
    the correct pattern and also not touched.

    Severity: P2 (silent, permanent, single-machine test skip -- not a
    crash, but a coverage gap that looks like coverage).
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        if not _is_test_file(py):
            continue
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if not _ABS_PATH_RE.match(node.value):
                continue
            findings.append(Finding(
                check="hardcoded_absolute_path_in_test",
                severity="P2",
                file=rel,
                line=node.lineno,
                snippet=_line_text(src_lines, node.lineno),
                detail=(
                    f"Hardcoded developer-machine path {node.value!r} in a test file -- this path exists on "
                    "exactly one machine; on any other, a fixture/skip-gate built on it silently and "
                    "permanently no-ops the test there. Use a relative/tmp_path/env-var-derived path instead."
                ),
            ))
    return findings
