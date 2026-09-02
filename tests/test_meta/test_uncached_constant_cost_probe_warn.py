"""Meta-test (WARN-ONLY): constant-cost work repeated on every call.

A helper shells out to a subprocess, rebuilds a ``ctypes`` library handle, or calls
``os.makedirs(exist_ok=True)`` on every call, from a dispatch path taken thousands of times. The
answer cannot change within a process, so every repetition after the first is pure overhead -
measured at 64 ms per dispatch decision for one nvidia-smi shell-out.

This module NEVER fails. It runs ``scan_uncached_constant_cost_probe`` over pyutilz's own source
and reports what it found as a ``UserWarning`` plus a terminal report, so the findings land in the
pre-commit/CI output as a periodic performance-triage list rather than as a commit gate. The
scanner is registered ``OPT_IN_ONLY``, so ``run_all()``'s default sweep - whose findings DO block
via ``test_code_audit_baseline.py`` - never picks it up by accident.

Warn rather than block because a probe that MUST be fresh - a liveness check, a config reload - is
indistinguishable from one that must not. The rule names the function and the call inside it;
whether caching is correct there stays a human decision.

The rule itself is unit-tested against reconstructed defects in
``tests/test_uncached_constant_cost_probe.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pyutilz
from pyutilz.dev.code_audit import scan_uncached_constant_cost_probe

SRC_DIR = Path(pyutilz.__file__).resolve().parent


def test_uncached_constant_cost_probe_advisory_report(capsys) -> None:
    """Report, never fail. An empty result is a clean report, not a skipped one."""
    findings = scan_uncached_constant_cost_probe(SRC_DIR)
    if findings:
        header = f"uncached_constant_cost_probe: {len(findings)} advisory finding(s)"
        report = "\n".join([header] + [f"  {f.file}:{f.line}  {f.detail}" for f in findings])
        warnings.warn(report, UserWarning, stacklevel=1)
        with capsys.disabled():
            print("\n" + report)
