"""Meta-test: run pyutilz.dev.code_audit's own scanners against pyutilz's
own source (self-scan), baseline-driven per this directory's snapshot-style
meta-test convention (see test_no_bare_except.py / test_no_mutable_defaults.py).

Findings are baselined together (keyed by ``check::file:line``) so
pre-existing debt doesn't block adoption -- only a NEW finding fails the
test. Refresh with ``--refresh-code-audit-baseline`` after a deliberate
change, or add a narrow, commented exclusion in the ``exclude_dirs``
passed below for a confirmed false positive.

This is the same baseline-driven wiring rolled out to every downstream
consumer of dev.code_audit (glossum_backend_scripts, llm_bench,
realtime_applications, production_scrapers, mlframe, algopacksimple) --
pyutilz eats its own dog food, and (like every other consumer) uses the
shared harness in py_ci_shared.code_audit_meta rather than a hand-rolled
copy. See ``pyutilz/src/pyutilz/dev/code_audit/__init__.py`` for what
each check catches.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.code_audit_meta import assert_no_new_code_audit_findings

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_baseline.json"


def test_no_new_code_audit_findings():
    assert_no_new_code_audit_findings(
        root=PYUTILZ_DIR,
        baseline_path=_BASELINE_PATH,
    )
