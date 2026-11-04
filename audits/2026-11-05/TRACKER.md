# 2026-11-05 py-ci-shared adoption: tracker

Gates found real hits but are not wired yet. None is baselined; each stays out of the meta tests until fixed.

| Status | Sev | ID | Title |
|---|---|---|---|
| **OPEN** | P3 | `G-01` | `patch_target_read_at_call`: 38 test patches of a name the module only re-exports, so the patch changes nothing (e.g. tests/llm/test_openrouter_coverage_20260926.py:65). |
| **OPEN** | P3 | `G-02` | `wallclock_test_waits`: 35 tests sleep on the wall clock (tests/test_monitoring.py, tests/test_single_flight_cache.py, ...). |
| **OPEN** | P3 | `G-03` | `disposition_test_references`: 46 audit dispositions under audits/ name tests or test files that no longer exist. |
| **OPEN** | P3 | `G-04` | `workflow_runner_labels`: 22 `*-latest` runner labels in .github/workflows; pinning changes the CI image matrix and needs a CI run to verify. |
