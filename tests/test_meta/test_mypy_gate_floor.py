"""The mypy_gate `--min-files` floor is derived from the measured source count, and both venues carry the same one.

A hand-set floor drifts: this repo's stood at 200 against a tree that had grown to 266 files, so a run that silently lost a
quarter of its scope still passed. The floor must sit between 90% and 100% of the files mypy checks today (every `.py`
under src/pyutilz); when the tree grows or shrinks past that band, re-measure (`python -m py_ci_shared.mypy_gate src/pyutilz`
prints the count) and move both copies together.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_FLOOR = re.compile(r"py_ci_shared\.mypy_gate --min-files (\d+) src/pyutilz")
_VENUES = (".pre-commit-config.yaml", ".github/workflows/mypy-full.yml")


def test_both_venues_run_mypy_gate_with_one_floor_sized_from_the_tree():
    """Pre-commit and CI name the same floor, within 90-100% of the current source-file count."""
    floors = {venue: [int(m) for m in _FLOOR.findall((REPO_ROOT / venue).read_text(encoding="utf-8"))] for venue in _VENUES}
    assert all(floors.values()), f"a venue no longer runs mypy_gate with --min-files over src/pyutilz: {floors}"
    values = {v for found in floors.values() for v in found}
    assert len(values) == 1, f"the venues disagree on the mypy floor: {floors}"
    floor = values.pop()
    measured = len(list((REPO_ROOT / "src" / "pyutilz").rglob("*.py")))
    assert 0.9 * measured <= floor <= measured, f"--min-files {floor} is not within 90-100% of the {measured} source files; re-measure and move both copies"
