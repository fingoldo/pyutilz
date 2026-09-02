"""Meta-test: run pyutilz.dev.code_audit's own scanners against pyutilz's
own source (self-scan), baseline-driven per this directory's snapshot-style
meta-test convention (see test_no_bare_except.py / test_no_mutable_defaults.py).

Findings are baselined together (keyed by ``check::file:line``) so
pre-existing debt doesn't block adoption -- only a NEW finding fails the
test. Refresh with ``--refresh-code-audit-baseline`` after a deliberate
change, or add a narrow, commented exclusion in the ``exclude_dirs``
passed below for a confirmed false positive.

2026-09-03 baseline refresh, reviewed individually rather than bulk-accepted. Eleven new
scanners were registered in this commit. They produced nine entries here; two of the nine were
real weaknesses in a new scanner and were fixed in it rather than baselined, leaving seven.

* `effect_flag_outside_its_effect` x2 -- FIXED, not baselined. `self._ready = True` after an
  unrelated `if self._process.stdout:` was reported purely because both statements mention
  `self`, and `res.add(str(obj))` after `if verbose: logger.info(...)` was reported because a
  logging-only guard was mistaken for the work being recorded. The scanner now drops `self`/`cls`
  as linking tokens and skips guards whose body only logs; both narrowings have a test that
  fails when the narrowing is removed.

* `comment_names_missing_symbol` x2, in `duplicate_function_body.py` and `vacuous_matching.py`.
  Both cite a helper (`_need_cuda()`, `_covers()`) from the codebase whose defect the comment
  DESCRIBES, not from this tree -- illustrative prose, not a rotted pointer. That distinction is
  not statically decidable, and it is a class this package will keep producing precisely because
  documenting defects found elsewhere is what these modules are for. Baselined rather than
  weakening the rule for every consumer, and rather than rewriting correct documentation to
  satisfy a linter.

* `default_via_or` x3 in the new scanners. All three are correct idioms, not traps:
  `calls.get(name, False) or <bool>` is a boolean OR on booleans, and `a.asname or a.name` is
  the canonical ast-alias idiom where `asname` is `str | None` and never the empty string.

* `unthrottled_hot_loop_log` x2 in `web/browser.py` -- the same two entries already baselined,
  moved by the ten comment lines this commit adds above them. Line shift, not a new finding.

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

import pytest

# py-ci-shared requires python>=3.9 (pyproject.toml dev-dependency marker), so it's not
# installed on the 3.8 CI leg -- skip cleanly there instead of erroring at collection.
py_ci_shared_code_audit_meta = pytest.importorskip("py_ci_shared.code_audit_meta")
assert_no_new_code_audit_findings = py_ci_shared_code_audit_meta.assert_no_new_code_audit_findings

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_baseline.json"


def test_no_new_code_audit_findings(request):
    assert_no_new_code_audit_findings(
        root=PYUTILZ_DIR,
        baseline_path=_BASELINE_PATH,
        request=request,
    )
