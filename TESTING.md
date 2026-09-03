# Testing

Test counts and coverage percentages are deliberately not pinned in this file — they drift with
every commit and go stale silently. Run `pytest --collect-only -q` for the current count; the
codecov badges in [README.md](README.md) carry the current line coverage. Live LLM-provider
tests are gated behind `--run-live` and skip by default so CI never
spends real money.

Runtimes below were measured on one Windows developer box (2026-09-02) and are indicative only —
they scale with core count and disk speed.

## Running tests

```bash
pytest                                          # full suite
pytest tests/test_meta/                          # static meta-tests only, ~2 min
pytest tests/test_pandaslib.py -v                # one module
pytest --run-live -m live                        # live LLM smoke tests (real API calls)
pytest -m "not slow"                             # fast inner loop: drops the multi-second tests
pytest --cov=src/pyutilz --cov-report=term-missing
```

**On Windows, add `--no-cov` to every local run.** pytest-cov raises `PermissionError` when it
writes its data file here; coverage is measured in CI on `ubuntu-latest`, so the `--cov` recipe
above is the CI form. If you do need a local number, `python -m coverage run -m pytest --no-cov`
followed by `python -m coverage report` works, because it keeps coverage out of the pytest
plugin path entirely.

Coverage is uploaded to Codecov on every CI run. The CI gate is `--cov-fail-under`; the same
floor is mirrored as `fail_under` under `[tool.coverage.report]` in `pyproject.toml` so a local
`coverage report` enforces it too. Treat it as a ratchet: raise it when coverage rises, never
lower it to make a run pass.

`slow` marks the tests that spend seconds sleeping or driving real timeouts rather than doing
work -- deliberate `time.sleep` budgets, timeout-saturation loops, and child-interpreter spawns.
They stay in the default run (nothing is skipped by default); `-m "not slow"` is the documented
way to drop them while iterating. The `gpu` marker is deselected in CI (no GPU runner) and
`live` is skipped unless `--run-live` is passed.

## Static meta-tests

`tests/test_meta/` is a static-check suite catching package-level
drift without exercising runtime behaviour. Wired into
`.pre-commit-config.yaml`, so configuration regressions are caught at
commit time. Selected entries:

| Test                                    | Polices                                                                                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_provider_registration.py`         | Every canonical name in `llm.factory._PROVIDER_MODULES` resolves; every alias has a target; no key collisions.                                       |
| `test_module_alias_integrity.py`        | The 27-entry backward-compat module alias map imports cleanly and proxies real symbols.                                                              |
| `test_provider_contract.py`             | Every concrete LLM provider inherits from `LLMProvider`, overrides every abstract method, and signature-matches the base interface.                  |
| `test_optional_deps_isolation.py`       | `import pyutilz` succeeds with each optional-dep group masked; sub-process isolated.                                                                 |
| `test_no_top_level_side_effects.py`     | Importing pyutilz performs zero network I/O at module-load time. Sub-process socket block.                                                           |
| `test_api_stability.py`                 | Snapshots the public surface (top-level `__all__`, alias map, public symbol set with signatures, class MROs). Renames / removals fail.               |
| `test_resource_handle_safety.py`        | Every `open()` / `Popen()` / `NamedTemporaryFile()` call is context-managed.                                                                         |
| `test_encoding_consistency.py`          | Every builtin `open(...)` in production code passes `encoding=` (Windows cp1251 safety).                                                             |
| `test_no_unicode_in_console_output.py`  | Snapshot-based check for non-ASCII string literals in `print(...)` / `logger.*(...)` calls (Windows stdout safety).                                  |
| `test_provider_cache_concurrency.py`    | 20 concurrent `get_llm_provider()` callers share one instance; constructor runs exactly once.                                                        |
| `test_no_import_cycles.py`              | Tarjan's SCC over the AST-built import graph; flags multi-node cycles.                                                                               |
| `test_logger_lazy_formatting.py`        | Logger calls use `%`-style formatting (lazy) instead of f-strings (eager) so messages aren't formatted when level is disabled.                       |
| `test_deferred_drift.py`                | Counts every `_USER_DEFERRED_*` whitelist across the meta-test suite. Fails when a whitelist grows; refresh via `--refresh-debt-baseline`.           |
| `test_shared_checks_wired.py`           | Runs the cross-project checks py-ci-shared already ships: 1000-LOC module budget over `src/` **and** `tests/`, per-job CI `timeout-minutes`, reviewed `continue-on-error`, entry-point resolvability, markdown-link targets, git-dependency pinning, CI reachability of every `tests/` subdir. |
| `test_scanner_positive_and_negative_cases.py` | Every registered `code_audit` scanner has both a test asserting a non-empty result and one asserting an empty result; exemptions need a written reason and go stale loudly. |
| `test_code_audit_tests_baseline.py`      | Points the TEST-QUALITY scanners at `tests/` itself — `nondiscriminating_test` and `source_text_assertion` included, so a test that asserts nothing, or asserts against source text instead of behaviour, fails here. Every baseline entry carries a written per-entry justification in the module docstring. |
| `test_no_module_reload.py`               | Bans unreviewed `importlib.reload()`, `sys.modules.pop("pyutilz...")` and bare `sys.modules[...] = ...` writes under `tests/` — the three routes that split module identity or leave a stub package visible to the rest of the process. |
| `test_gate_integrity.py`                | Every blocking gate's scope narrowing carries a written reason; mypy runs through a wrapper that requires its completion line, not just exit 0; the CI coverage floor equals `[tool.coverage.report] fail_under`. |
| `test_prose_numeric_claims.py`          | Counted facts in prose (alias count, provider count, mypy source-file count) are computed, not typed. Rewording a claim's anchor is itself a failure. |
| `test_docs_inventory_parity.py`         | Extras-group descriptions match `[project.optional-dependencies]`; documented paths and pytest markers exist; every shipped module appears in some orientation doc (warn). |

Each meta-test exposes one or both whitelists at file scope:

- `_KNOWN_*` — items consumed via routes static analysis can't see; cite
  the consumer location.
- `_USER_DEFERRED_*` — items the maintainer surfaced and chose to defer
  cleanup on. Drain to zero over time.

Shared helpers (`consumer_corpus`, `public_top_level_symbols`,
`capture_signature`, `count_user_deferred_entries`, etc.) live in
[`pyutilz.dev.meta_test_utils`](src/pyutilz/dev/meta_test_utils.py).

## Live LLM tests

Live tests (`tests/test_llm_live.py`) hit real provider APIs and cost a
fraction of a cent per run. Setup:

1. Copy `.env.example` to `.env` and fill in the keys you have. Per-provider
   fixtures skip individually when a key is missing, so contributors with
   a subset of accounts still get partial coverage.
2. Run `pytest --run-live tests/test_llm_live.py`. Each test asserts
   `assert_under_budget` ($0.005 cap by default) so an accidental huge
   prompt fails the test rather than burning credits.

`.env` is gitignored; the [detect-secrets](https://github.com/Yelp/detect-secrets)
pre-commit hook blocks accidental commits of API keys to source files.

## Test layout

Most test files sit flat at `tests/` root; four subfolders exist for a real reason:
`tests/code_audit/`, `tests/performance/kernel_tuning/`, `tests/stats/`, and `tests/system/` group tests for source
subpackages that were split into multiple files (`performance/kernel_tuning/*`, `stats/*`,
`system/gpu_dispatch.py` + `system/system/*`) — one test-file-per-source-file would otherwise
scatter closely-related coverage across many flat-root files. New tests for those three source
areas go in the matching subfolder; everything else (including the rest of `system/` —
`monitoring.py`, `parallel.py`, `distributed.py` — which predate this convention and stayed at
flat-root for historical reasons) goes at `tests/` root as `test_<module_name>.py`.

`tests/code_audit/` mirrors `src/pyutilz/dev/code_audit/` one-for-one: one `test_<family>.py` per
scanner family, named after the source module it exercises (`test_mutable_defaults.py` for
`mutable_defaults.py`), plus a few files named after an audit wave rather than a scanner
(`test_audit_20260903_scanner_fixes_a_l.py`) where the wave's fixes landed as one reviewed set and
each test names the finding it pins. It was carved out of a single 11306-line flat-root
test_code_audit file — the production side of the same feature had already been split into
one module per scanner to respect CLAUDE.md's module-size rule, while the test side had not.
Fixtures used by more than one family (`_write`, and the snippet builders shared across waves)
live in `tests/code_audit/_helpers.py`; a new scanner gets a new `test_<family>.py`, never an
append to an unrelated one.

Two meta-tests hold that structure in place:

- `tests/test_meta/test_shared_checks_wired.py::test_no_new_file_over_1k_loc` applies the
  1000-LOC module budget to `tests/` as well as `src/`. The files already over the limit are
  grandfathered at their measured size in `_loc_over_1k_baseline.json` with zero growth slack, so
  they may shrink but not grow, and no new oversized file may be added on either side. Refresh
  the baseline (`--refresh-loc-budget-baseline`) only after a real split or shrink.
- `tests/test_meta/test_scanner_positive_and_negative_cases.py` enforces, for every scanner in
  `get_scanners()`, that some test asserts a NON-empty result (the scanner fires on the defect)
  and some test asserts an EMPTY one (it stays silent on clean input). Both are read statically
  out of the test sources. A scanner that genuinely cannot have a clean case goes in that file's
  `_NO_CLEAN_CASE_POSSIBLE` with a written reason; the dict is empty today, and a third test
  fails if an entry there goes stale.

`_extra`/`_extra2`/`_extra3` suffixes on some files (e.g. `test_system_extra3.py`,
`test_pandaslib_extra2.py`) mark tests added later to close a specific coverage gap in the base
`test_<module>.py` file for the same module — check the base file first for existing coverage of
a function before adding a new test, regardless of which numbered file it ends up in.

Three separate "benchmark" surfaces exist, easy to confuse by name: `_benchmarks/bench_*.py`
(standalone scripts, `python -m _benchmarks.bench_*`, never run by pytest/CI — hard-coded
identity assertions comparing an old vs. new implementation) vs. `tests/test_kernel_tuning_benchmark.py`
(real, CI-collected pytest coverage for `pyutilz.performance.kernel_tuning.benchmark`) vs.
`tests/test_dev_benchmarking.py` (real, CI-collected pytest coverage for the unrelated
`pyutilz.dev.benchmarking` module). A fourth name looks like a surface and is not one: the
`pytest-benchmark` plugin is declared in `[dev]` and loads on every run (it appears in the pytest
header), but no test uses its `benchmark` fixture -- a `test_*_benchmark` name in `tests/` always
belongs to one of the three surfaces above, never to that plugin.

## Pre-commit hook

```bash
pip install pre-commit vulture && pre-commit install
```

The hook runs the meta-test suite on every commit — measured at ~2 min on the reference box
(2026-09-02), not the seconds-scale cost the name suggests. For tight inner-loop work, a
`manual`-stage variant skips the two sub-process-spawning tests
(`test_optional_deps_isolation.py`, `test_no_top_level_side_effects.py`) and roughly halves that,
to ~1 min:

```bash
pre-commit run --hook-stage=manual pyutilz-meta-tests-static-only
```

If a full meta-test run is too slow for your loop, run that manual variant while iterating and
let the blocking pre-commit hook do the full pass at commit time — `--no-verify` skips every
other gate too (Black, ruff, bandit, vulture, deptry) and is not the cheaper option it looks like.

## Local pip-audit on Windows

`pip-audit --desc` (the default text formatter) crashes with `UnicodeEncodeError`
on a Windows console using the cp1251 codepage -- some CVE descriptions contain
Unicode arrows the console can't encode. No GitHub Actions workflow runs pip-audit at all --
its only wiring in this repo is the advisory, warn-only `.pre-commit-config.yaml` hook -- so a
developer's console is the only place it ever runs, and on this project's reference box that is
the Windows console this note is about. Run it locally, writing JSON instead of printing text:

```bash
pip-audit --desc -f json -o pip-audit-report.json
```
