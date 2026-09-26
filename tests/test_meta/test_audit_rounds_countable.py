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
    "`src/pyutilz/dev/code_audit/` - 59 scanner modules",
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


def test_rounds_are_filed_where_their_state_says():
    """A fully closed round lives in audits/implemented/, and a filed round has no open row."""
    assert_rounds_filed(AUDITS, known=KNOWN_FILING_PROBLEMS, min_trackers=0)
