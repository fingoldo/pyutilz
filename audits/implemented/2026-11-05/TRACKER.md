# 2026-11-05 py-ci-shared adoption: tracker

Gates that found real hits when the e13229e pin was adopted. None is baselined.

| Status | Sev | ID | Title |
|---|---|---|---|
| **RESOLVED** | P3 | `G-01` | `patch_target_read_at_call`: the 38 hits are all patches of the `openrouter_provider` and `kernel_tuning.cache` facade packages. The code reads those names late through `sys.modules` / `_pkg()` on purpose (acyclic import graph), so the patches are live; the gate cannot see that lookup. Left unwired and reported to py-ci-shared as a gate defect. |
| **RESOLVED** | P3 | `G-02` | `wallclock_test_waits`: 35 sleeping tests fixed in code and the gate wired blocking. Simulated work became an injected clock (`dev.benchmarking.timer`, `monitoring.time`, the `timer=` argument of `benchmark_backends`) or event-loop turns; cancellation stand-ins block on an `Event`; thread-ordering sleeps became `join(timeout=...)`. |
| **RESOLVED** | P3 | `G-03` | `disposition_test_references`: 46 dispositions named tests that no longer exist. Each reference now names where the test went (renames of 2026-10-24/26/27, the code_audit tests moved to py-ci-shared on 2026-10-22, downstream-repo tests) or is plain prose; no finding row was removed. Gate wired blocking. |
| **RESOLVED** | P3 | `G-04` | `workflow_runner_labels`: the 22 `*-latest` labels are pinned with the gate's own `--fix` (ubuntu-24.04, windows-2025, macos-15; the 3.8/3.9 floor legs keep ubuntu-22.04 and macos-15-intel, as the matrix already chose). Gate wired blocking. |
