# pyutilz audit — round 2 (cross-cutting, new angles)

10 parallel read-only agents, each auditing a lens not covered by round 1 (see `audits/2026-07-21_full-audit/`) and not caught by the 26 automated `pyutilz.dev.code_audit` scanners added since round 1 (`undeclared_import`, `vacuous_assertion`, `locals_globals_as_output`, `missing_network_timeout`, `parameter_aliasing_mutation`, `sync_blocking_in_async`, `retry_loop`, `duplicate_module_docstring`, `unraised_exception_class`, `credential_shaped_log_arg`, `docstring_args_incomplete`, `return_annotation_mismatch`, plus the 14 from round 1). Severity counts below are grep-verified against each report's actual `### [SEVERITY]` headers, not agent self-reported prose.

## Totals across all 10 reports

| Severity | Count |
|---|---|
| Critical | 2 |
| High | 26 |
| Medium | 21 |
| Low | 9 |
| **All findings** | **58** |

## Reports (grep-verified severity counts)

| # | Report | Critical | High | Medium | Low |
|---|---|---|---|---|---|
| 01 | [Concurrency & Thread/Async Safety](01-concurrency-async-safety.md) | 1 | 2 | 1 | 0 |
| 02 | [Resource Lifecycle & Leaks](02-resource-lifecycle-leaks.md) | 0 | 3 | 0 | 0 |
| 03 | [Numerical & Data Correctness](03-numerical-data-correctness.md) | 1 | 3 | 1 | 1 |
| 04 | [API Design Consistency & Ergonomics](04-api-design-consistency.md) | 0 | 4 | 4 | 2 |
| 05 | [Caching & Memoization Correctness](05-caching-memoization.md) | 0 | 1 | 3 | 2 |
| 06 | [Configuration & Cross-Platform Robustness](06-config-cross-platform.md) | 0 | 3 | 0 | 0 |
| 07 | [Logging & Observability Quality](07-logging-observability.md) | 0 | 4 | 5 | 1 |
| 08 | [Test Suite Behavioral Coverage Gaps](08-test-coverage-gaps.md) | 0 | 4 | 2 | 1 |
| 09 | [Backward-Compatibility & Versioning Risk](09-backward-compat-versioning.md) | 0 | 1 | 3 | 2 |
| 10 | [Performance & Algorithmic Complexity](10-performance-complexity.md) | 0 | 1 | 2 | 0 |

## Next step

Triage: fix now vs. defer vs. dismiss, starting with the 2 Critical items.
