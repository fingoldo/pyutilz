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

Refresh the baseline with ``--refresh-code-audit-baseline`` after REVIEWING each new entry;
every finding currently in ``_code_audit_tests_baseline.json`` was read and judged a false
positive or an accepted precondition (see the notes in that review, summarised here):

* ``except_skip_masks_call_under_test`` x4 -- ``test_llm_live.py:182`` catches only
  ``asyncio.TimeoutError`` (an infinite tenacity retry against a rate-limited key),
  ``test_llm_live.py:205`` re-``raise``s anything that is not an upstream 503/quota message,
  ``test_pandaslib_extra2.py:248`` is the dtype-availability helper whose ``except`` covers the
  ``pd.array(...)`` construction only (the function under test is called outside it), and
  ``test_strings_extra2.py:502`` covers ``spacy.load()`` of an undownloaded model. The scanner
  sees the ``try`` but not the narrowing.
* ``vacuous_assertion`` x2 -- ``system/test_hardware_detection.py:257``/``:289`` assert
  ``result is None or isinstance(result, dict)`` for the power-plan / battery probes, which is
  the correct contract on a desktop with no battery; the shape assertions those functions
  really need are made unconditionally against a monkeypatched psutil in the sibling tests.
* ``vacuous_empty_pattern_match`` x2 -- both are preceded by an explicit non-empty assertion
  (``assert reported`` / ``assert seen``) that the scanner only recognises in ``len(...) ==``
  form.
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
