"""Meta-test: run pyutilz.dev.code_audit's own scanners against pyutilz's
own source (self-scan), baseline-driven per this directory's snapshot-style
meta-test convention (see test_no_bare_except.py / test_no_mutable_defaults.py).

Findings are baselined together (keyed by ``check::file:line``) so
pre-existing debt doesn't block adoption -- only a NEW finding fails the
test. Refresh with ``--refresh-code-audit-baseline`` after a deliberate
change, or add a narrow, commented exclusion in ``_EXCLUDE_DIRS`` for a
confirmed false positive.

This is the same baseline-driven wiring rolled out to every downstream
consumer of dev.code_audit (glossum_backend_scripts, llm_bench,
realtime_applications, production_scrapers, mlframe) -- pyutilz eats its
own dog food. See ``pyutilz/src/pyutilz/dev/code_audit/__init__.py`` for
what each check catches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import pytest

import pyutilz
from pyutilz.dev.code_audit import Finding, run_all

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_baseline.json"

_EXCLUDE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules",
})


def _refresh_requested() -> bool:
    return "--refresh-code-audit-baseline" in sys.argv


def _key(f: Finding) -> str:
    return f"{f.check}::{f.file}:{f.line}"


def _current_findings() -> list[Finding]:
    return run_all(PYUTILZ_DIR, exclude_dirs=_EXCLUDE_DIRS)


def test_no_new_code_audit_findings():
    current = _current_findings()
    current_by_key = {_key(f): f for f in current}
    current_keys = set(current_by_key)

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            orjson.dumps(sorted(current_keys), option=orjson.OPT_INDENT_2).decode("utf-8"),
            encoding="utf-8",
        )
        pytest.skip(f"code-audit baseline refreshed at {_BASELINE_PATH.name} " f"({len(current_keys)} existing finding(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current_keys - baseline)
    fixed = sorted(baseline - current_keys)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_code_audit_findings] {len(fixed)} finding(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-code-audit-baseline\n"
        )

    if new:
        detail_lines = []
        for k in new[:20]:
            f = current_by_key.get(k)
            detail_lines.append(f"{f.check} [{f.severity}] {f.file}:{f.line} -- {f.detail}" if f is not None else k)
        pytest.fail(
            f"{len(new)} new static-analysis finding(s) from "
            f"pyutilz.dev.code_audit (see "
            f"pyutilz/src/pyutilz/dev/code_audit/__init__.py for check "
            f"descriptions). Fix the code, OR if this is a confirmed false "
            f"positive, refresh the baseline after review:\n  " + "\n  ".join(detail_lines) + (f"\n  ... and {len(new) - 20} more" if len(new) > 20 else "")
        )
