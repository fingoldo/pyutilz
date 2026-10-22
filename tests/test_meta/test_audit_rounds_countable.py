"""Audit rounds under audits/ stay countable by machine (py_ci_shared.audit_round_format).

A round is only closed if it can be counted as closed: every finding heading is a finding id with a disposition, and a
round sits in audits/implemented/ exactly when nothing in it is open. pyutilz numbers findings per report file (`### F01.`
restarts in every file, `### MT-3.` in the preventive-meta-tests report), so ids are file-scoped and the repo-wide "one id,
one heading" half of ``assert_rounds_countable`` does not apply here; each file is checked on its own instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from py_ci_shared.audit_round_format import assert_rounds_filed, finding_problems, round_files

AUDITS = Path(__file__).resolve().parents[2] / "audits"
# Older rounds number findings ``F12.`` / ``MT-3.``; rounds from 2026-09-26 on use the shared ``OR-1 (High)`` form.
FINDING_ID_RE = re.compile(r"^((?:F|MT-)\d+(?=\.)|[A-Z]+-\d+(?= \())")

# Section headings of the preventive-meta-tests report's inventory, prose rather than findings.
NOT_FINDINGS = (
    "`tests/test_meta/` - 40 test modules, 15 baseline JSON files",
    "`src/py_ci_shared/code_audit/` - 59 scanner modules",
    "`py-ci-shared/src/py_ci_shared/` - 21 modules, 1 consumed by pyutilz's meta-suite",
    "`.pre-commit-config.yaml` - about 15 hooks, most blocking",
)

# The two 2026-07-21 rounds predate trackers and in-file dispositions, so nothing records which of their findings closed.
# Shrink-only: give a round a TRACKER.md with a status per finding (and file it once closed), then drop its entry.
KNOWN_FILING_PROBLEMS = (
    "2026-07-21_audit-round2: open round with no TRACKER*.md - its closure cannot be counted",
    "2026-07-21_full-audit: open round with no TRACKER*.md - its closure cannot be counted",
)


def test_every_finding_in_a_dispositioned_round_has_a_disposition():
    """Each finding heading is an id with a `**Disposition**` line; continuation headings go at `####`."""
    files = round_files(AUDITS)
    assert len(files) >= 20, f"only {len(files)} round files under {AUDITS}; the layout moved or the glob broke"
    problems: list[str] = []
    checked = 0
    for path in files:
        found = finding_problems(path, finding_id_re=FINDING_ID_RE, not_findings=NOT_FINDINGS)
        if found is not None:
            checked += 1
            problems.extend(f"{path.parent.name}/{p}" for p in found)
    # 11 + 10 report files in the two filed rounds keep dispositions; fewer means the disposition pattern stopped matching.
    assert checked >= 20, f"only {checked} round files keep dispositions; nothing much was checked"
    assert not problems, "findings that cannot be counted:\n  " + "\n  ".join(problems)


def _remove_empty_leftover_rounds(audits: Path) -> list[str]:
    """Remove empty round folders left behind by a move. Returns the ones that could not be removed (locked on Windows)."""
    stuck = []
    for folder in sorted(audits.iterdir()):
        if folder.is_dir() and folder.name != "implemented" and not any(folder.iterdir()):
            try:
                folder.rmdir()
            except OSError as exc:
                stuck.append(f"{folder.name} ({exc.strerror or exc})")
    return stuck


def test_rounds_are_filed_where_their_state_says():
    """A fully closed round lives in audits/implemented/, and a filed round has no open row.

    An empty folder left behind by a move is removed here first. If Windows still holds it open, it is skipped and
    named as a known filing problem, with a warning, instead of failing the gate; it is removed on the next run once
    the lock clears (the known entry is then stale and the gate asks for it to be dropped).
    """
    import warnings

    stuck = _remove_empty_leftover_rounds(AUDITS)
    known = list(KNOWN_FILING_PROBLEMS)
    for name in stuck:
        warnings.warn(f"empty audit round folder {name} is still held by another process; skipped, remove it when the lock clears", stacklevel=1)
        known.append(f"{name.split(' ')[0]}: open round with no TRACKER*.md - its closure cannot be counted")
    assert_rounds_filed(AUDITS, known=tuple(known), min_trackers=0)
