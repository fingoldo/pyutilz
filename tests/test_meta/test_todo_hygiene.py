"""PT-4 — TODO/FIXME/XXX/HACK hygiene meta-test.

Every marker in production code under ``src/pyutilz/`` must carry an
attribution: assignee in parens (``TODO(name): ...``) or ISO date
(``TODO 2026-04-28: ...``) or ``@assignee`` mention.

Uses the shared scanner / regex from ``pyutilz.dev.meta_test_utils`` so
the same logic is portable across projects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import pyutilz
from pyutilz.dev.meta_test_utils import (
    ATTRIBUTION_RE,
    scan_todo_markers,
)

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent

_MARKERS = ("TODO", "FIXME", "XXX", "HACK")

# Grandfathered un-attributed markers — drain over time. Each entry is
# "rel/path.py:lineno".
_GRANDFATHERED: set[str] = {
    # ``meta_test_utils.py`` documents marker keywords in section
    # headers and docstrings — pattern-matching tooling, not actual
    # work-items. Excluded permanently.
    "dev/meta_test_utils.py:251",
    "dev/meta_test_utils.py:275",
}


def _scan_markers() -> list[tuple[Path, int, str, str]]:
    return scan_todo_markers(PYUTILZ_DIR)


def test_every_todo_marker_has_attribution():
    bare: list[str] = []
    for path, lineno, kw, line in _scan_markers():
        rel = path.relative_to(PYUTILZ_DIR).as_posix()
        ident = f"{rel}:{lineno}"
        if ident in _GRANDFATHERED:
            continue
        if not ATTRIBUTION_RE.search(line):
            bare.append(f"{ident}  {line[:100]}")

    if bare:
        pytest.fail(
            f"{len(bare)} {'/'.join(_MARKERS)} comment(s) without "
            f"attribution. Add an assignee in parens (``TODO(name): ...``) "
            f"or an ISO date (``TODO 2026-04-28: ...``). To grandfather, "
            f"list ``<path>:<lineno>`` in _GRANDFATHERED:\n  "
            + "\n  ".join(bare[:30])
            + (f"\n  ... and {len(bare) - 30} more" if len(bare) > 30 else "")
        )


def test_todo_marker_count_summary_warning():
    """Warning-only — print marker totals so the curation prompt stays
    visible."""
    counts: dict[str, int] = {kw: 0 for kw in _MARKERS}
    by_file: dict[str, int] = {}
    total = 0
    for path, _, kw, _ in _scan_markers():
        counts[kw] += 1
        rel = path.relative_to(PYUTILZ_DIR).as_posix()
        by_file[rel] = by_file.get(rel, 0) + 1
        total += 1
    if total == 0:
        return
    top = sorted(by_file.items(), key=lambda kv: -kv[1])[:5]
    sys.stderr.write(
        f"\n[test_todo_marker_count_summary_warning] {total} marker(s) "
        f"across {len(by_file)} file(s); breakdown: "
        + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        + ". Top files: "
        + ", ".join(f"{p} ({n})" for p, n in top)
        + "\n"
    )
