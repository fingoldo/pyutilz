"""Meta-test (WARN-ONLY): per-call state stored on a shared provider instance.

A class whose instances are cached and handed to many callers stores what the LAST call produced
(``last_usage``, ``_last_json_schema_applied``, ``last_generation_id``) on ``self``, and a summary
accessor reads it back. Two concurrent calls interleave and the accessor reports the other
request's numbers. Nothing raises; the values are simply attributed to the wrong call. The same
defect landed in two consecutive audit waves on the same surface, which is why it has a check.

This module NEVER fails. It runs ``scan_per_call_state_on_shared_instance`` over pyutilz's own
source and reports what it found as a ``UserWarning`` plus a terminal report, so the findings land
in the pre-commit/CI output as a triage list rather than as a commit gate. The scanner is
registered ``OPT_IN_ONLY``, so ``run_all()``'s default sweep - whose findings DO block via
``test_code_audit_baseline.py`` - never picks it up by accident.

Warn rather than block because lock detection is lexical here: a lock taken by the CALLER reads as
absent, and a provider that is single-caller by contract is a legitimate exception. Those are
judgements to make per finding, not conditions to fail a commit on.

The rule itself is unit-tested against reconstructed defects in
``tests/test_per_call_state_on_shared_instance.py``. Promote this to blocking by removing the
scanner from ``OPT_IN_ONLY`` and refreshing the code-audit baseline, once the list has stabilised.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pyutilz
from pyutilz.dev.code_audit import scan_per_call_state_on_shared_instance

SRC_DIR = Path(pyutilz.__file__).resolve().parent


def test_per_call_state_on_shared_instance_advisory_report(capsys) -> None:
    """Report, never fail. An empty result is a clean report, not a skipped one."""
    findings = scan_per_call_state_on_shared_instance(SRC_DIR)
    if findings:
        header = f"per_call_state_on_shared_instance: {len(findings)} advisory finding(s)"
        report = "\n".join([header] + [f"  {f.file}:{f.line}  {f.detail}" for f in findings])
        warnings.warn(report, UserWarning, stacklevel=1)
        with capsys.disabled():
            print("\n" + report)
