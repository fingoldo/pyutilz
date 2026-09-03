"""Meta-test: run pyutilz.dev.code_audit's TEST-QUALITY scanners against ``tests/`` itself.

Companion to :mod:`test_code_audit_baseline`, which scans ``src/pyutilz`` only. The 2026-09-02
test-suite audit (F01) found that the repo ships test-quality scanners with full unit coverage
and never points a single one of them at its own test suite -- so nine ``except Exception:
pytest.skip(...)`` sites that silently reclassified genuine API breaks as environment skips sat
unnoticed. pyutilz is the reference wiring every downstream consumer copies; its own suite
should not be the one directory exempt from it.

Scope is a deliberate SUBSET of the registry rather than every scanner, because most checks are
written for production code and misfire wholesale on test code:

* ``redundant_test_fit_call`` flags ~100 ordinary ``obj = _helper()`` lines in this suite.
* ``duplicate_function_body`` flags ~30 identical-by-design fixtures, ``_set_parents`` AST
  helpers and one-line parametrised bodies.
* ``bare_except`` / ``broad_except_swallow`` / ``mutable_default`` etc. already have their own
  dedicated meta-tests with their own baselines.

The checks kept below are the ones whose whole point is "this test does not actually test
anything" -- exactly the failure class the audit was about.

``nondiscriminating_test`` and ``source_text_assertion`` joined the list on 2026-09-03 (audit
F01/F02). Both are registered, neither is ``OPT_IN_ONLY``, and both were written specifically to
flag test code -- but the list is a hand-maintained literal, so each landed unwired and its
findings accumulated silently: 74 nondiscriminating and 8 source-text sites had built up by the
time the audit measured them. Wiring them drained 60 real defects (assertions added) and left
6 reviewed false positives, below. NOTE FOR THE NEXT SCANNER OF THIS CLASS: a new test-quality
check must be added to ``_TEST_QUALITY_CHECKS`` in the same change that registers it, or this
happens again -- the list does not derive itself from the registry.

Refresh the baseline with ``--refresh-code-audit-baseline`` after REVIEWING each new entry;
every finding currently in ``_code_audit_tests_baseline.json`` was read and judged a false
positive or an accepted precondition (see the notes in that review, summarised here). The notes
are keyed by check plus TEST FUNCTION NAME rather than by line number, because absolute line
numbers rot on every edit above the site -- which is exactly what had happened to three of them
by 2026-09-03 (audit F10):

* ``except_skip_masks_call_under_test`` x5 -- ``test_llm_live.py``'s
  ``test_generate_json_smoke`` catches only ``asyncio.TimeoutError`` (an infinite tenacity retry
  against a rate-limited key) and its sibling re-``raise``s anything that is not an upstream
  503/quota message; ``test_pandaslib_extra2.py``'s dtype-availability helper has an ``except``
  covering the ``pd.array(...)`` construction only (the function under test is called outside
  it); ``test_strings_extra2.py``'s spacy helper covers ``spacy.load()`` of an undownloaded
  model; and ``test_meta/test_complexity_ratchet.py``'s helper skips when ruff is absent from
  PATH or returns non-JSON, neither of which is the code under test. The scanner sees the
  ``try`` but not the narrowing.
* ``vacuous_assertion`` x2 -- ``system/test_hardware_detection.py``'s power-plan and battery
  probes assert ``result is None or isinstance(result, dict)``, which is the correct contract on
  a desktop with no battery; the shape assertions those functions really need are made
  unconditionally against a monkeypatched psutil in the sibling tests.
* ``vacuous_empty_pattern_match`` x7 -- every one is an ``all(...)`` preceded by an explicit
  non-empty assertion on the same collection (``assert reported`` / ``assert seen`` /
  ``assert creating`` / ``assert modes``), which the scanner only recognises in ``len(...) ==``
  form. The four in ``test_security_audit_20260902.py`` guard the ``os.open`` spy lists behind
  ``assert creating`` / ``assert modes`` messages that name what must have been created, so an
  empty spy list fails there before the ``all(...)`` is reached.
* ``nondiscriminating_test`` x6 -- five are the repo's deliberately advisory meta-tests
  (``test_docs_inventory_parity``'s WARN-only documentation-coverage report,
  ``test_prose_numeric_claims``' undated-figure warning, ``test_todo_hygiene``'s marker-count
  summary, and the two ``*_warn.py`` advisory reports whose module docstrings state outright
  "This module NEVER fails" and register their scanner ``OPT_IN_ONLY``). Each has a BLOCKING
  sibling that does the enforcing -- ``test_prose_numeric_claims_match_the_repo``,
  ``test_every_todo_marker_has_attribution``, ``tests/test_per_call_state_on_shared_instance.py``
  and ``tests/test_uncached_constant_cost_probe.py`` -- so the advisory half is a curation
  prompt, not the gate. The sixth, ``test_monitoring.py``'s ``test_function``, is not a test at
  all: it is a one-line helper nested inside
  ``TestTimeoutWrapper.test_multiple_calls_reuse_executor`` whose name matches the ``test_``
  prefix by accident; the enclosing test asserts ``results == list(range(10))`` over ten calls
  through it. The scanner does not descend into nested functions, so it scores the helper as a
  standalone test body. Renaming it purely to dodge the prefix match would be dodging the
  scanner, so it is baselined instead.
* ``tautological_is_not_none_only_test`` and ``source_text_assertion`` -- zero entries. Both are
  fully drained under ``tests/`` and must stay that way.
* ``vacuous_empty_pattern_match`` x2 more (``code_audit/test_domain_boundary.py``,
  ``code_audit/test_redundant_test_fit.py``) -- both ``all(...)`` calls are preceded by an
  assertion that already establishes a non-empty result (``[f.check for f in findings] == [...]``
  with two elements; ``"redundant_test_fit_call" in checks`` where ``checks`` is built from
  ``findings``), neither of which is one of ``vacuous_matching``'s guard spellings. They surfaced
  only when ``tests/test_code_audit.py`` was split into ``tests/code_audit/``: ``_guarded`` accepts
  a guard anywhere in the MODULE, and in an 11306-line file an unrelated test's ``len(findings) ==``
  was covering them. The split is what made the check honest here, not what broke these tests.
* ``hardcoded_absolute_path_in_test`` x4 -- ``is_local_path``'s whole job is classifying
  absolute Windows paths, and the ``psutil`` disk-partition mocks must carry a real mountpoint
  string; neither touches the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# py-ci-shared requires python>=3.9 (pyproject.toml dev-dependency marker), so it is not
# installed on the 3.8 CI leg -- skip cleanly there instead of erroring at collection.
py_ci_shared_code_audit_meta = pytest.importorskip("py_ci_shared.code_audit_meta")
assert_no_new_code_audit_findings = py_ci_shared_code_audit_meta.assert_no_new_code_audit_findings

TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_tests_baseline.json"

_TEST_QUALITY_CHECKS = [
    "except_skip_masks_call_under_test",
    "hardcoded_absolute_path_in_test",
    "nondiscriminating_test",
    "source_text_assertion",
    "stale_test_spy_arity",
    "tautological_is_not_none_only_test",
    "unenforced_docstring_invariant",
    "vacuous_assertion",
    "vacuous_empty_pattern_match",
]


def test_no_new_test_quality_findings_in_the_test_suite(request):
    assert_no_new_code_audit_findings(
        root=TESTS_DIR,
        baseline_path=_BASELINE_PATH,
        checks=_TEST_QUALITY_CHECKS,
        request=request,
    )
