# pyutilz full audit — 2026-07-21

13 parallel read-only agents (6 cross-cutting dimensions + 7 module-domain deep dives), each reading its scope in full and verifying findings empirically (reproduction scripts, differential fuzzing, actual installed-package checks) rather than by inference alone. Severity counts below are grep-verified against each report's actual `### [SEVERITY]` headers, not the agents' self-reported prose (a few agents miscounted their own totals by one).

## Totals across all 13 reports

| Severity | Count |
|---|---|
| Critical | 5 |
| High | 52 |
| Medium | 85 |
| Low | 62 |
| **All findings** | **204** |

## The 5 Critical findings

1. **`core/pythonlib.py` unconditionally imports numba/joblib/portalocker** — contradicts README's "zero hard dependencies" promise for the library's most-shared helper module. `src/pyutilz/core/pythonlib.py:31,37,869`. Flagged independently by both [01-architecture-design.md](01-architecture-design.md) and [03-packaging-dependencies.md](03-packaging-dependencies.md).
2. **Proxy credentials and Authorization headers logged in cleartext**, unconditionally, on a common code path. `src/pyutilz/web/web.py:382-385,503-506`. [02-security.md](02-security.md)
3. **Plaintext proxy credentials written to a predictable, non-temp, never-cleaned-up file.** `src/pyutilz/web/browser.py:220-255`. [02-security.md](02-security.md)
4. **`connect_to_s3()` never applies the credentials it reads — always authenticates with `None`/`None`.** `src/pyutilz/cloud/cloud.py:78-93`. [10-domain-web-cloud.md](10-domain-web-cloud.md)

(4 distinct Critical issues; #1 was independently found and reported by 2 agents.)

## Reports (grep-verified severity counts)

| # | Report | Critical | High | Medium | Low |
|---|---|---|---|---|---|
| 01 | [Architecture & design](01-architecture-design.md) | 1 | 4 | 4 | 2 |
| 02 | [Security (semantic/business-logic)](02-security.md) | 2 | 2 | 6 | 2 |
| 03 | [Packaging & dependencies](03-packaging-dependencies.md) | 1 | 9 | 3 | 1 |
| 04 | [Test suite quality](04-test-suite-quality.md) | 0 | 3 | 10 | 4 |
| 05 | [Documentation accuracy & quality](05-documentation.md) | 0 | 1 | 7 | 10 |
| 06 | [OSS / GitHub project health](06-oss-github-practices.md) | 0 | 1 | 4 | 7 |
| 07 | [Domain: core + dev (incl. code_audit)](07-domain-core-dev.md) | 0 | 3 | 13 | 8 |
| 08 | [Domain: data (pandaslib/polarslib/numpylib/numbalib/stats)](08-domain-data.md) | 0 | 2 | 5 | 3 |
| 09 | [Domain: database](09-domain-database.md) | 0 | 5 | 7 | 7 |
| 10 | [Domain: web + cloud](10-domain-web-cloud.md) | 1 | 12 | 8 | 4 |
| 11 | [Domain: system + performance](11-domain-system-performance.md) | 0 | 5 | 7 | 6 |
| 12 | [Domain: text](12-domain-text.md) | 0 | 2 | 5 | 3 |
| 13 | [Domain: llm](13-domain-llm.md) | 0 | 3 | 6 | 5 |

## Recurring cross-report themes

- **The "zero hard dependencies" / install-only-what-you-need promise is broken in multiple places**, not just `pythonlib.py`: reports 01 and 03 independently found ~8 modules with undocumented unconditional third-party imports, and 03 additionally found the existing `test_optional_deps_isolation` meta-test has a structural blind spot (only probes package `__init__.py` files, never leaf submodules) that let all of these slip past CI.
- **Web/cloud (report 10) is disproportionately loaded** (1 Critical, 12 High) — missing timeouts, retry loops with no backoff/idempotency guard, and resource leaks (`.close()` vs `.quit()`) cluster heavily in `web.py` and `browser.py`.
- **LLM provider retry logic (report 13) has real correctness gaps**: Gemini and Anthropic retry predicates don't match the exception hierarchies the installed SDKs actually raise, so some real 429/5xx errors are silently never retried.
- **The `code_audit` dogfooded lint tool (report 07) has concrete false-positive/false-negative gaps** in exactly the way its own docstrings claim it doesn't (e.g. `scan_dead_cli_flags`'s "never a false positive" claim disproven with a reproduction).

## Next step

Go through this list together and triage: fix now vs. defer vs. dismiss, starting with the 5 Critical items.
