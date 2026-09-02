# Test Suite Quality Audit — pyutilz (2026-09-02)

## Summary

Read `CLAUDE.md`, `TESTING.md`, `audits/2026-07-21_full-audit/04-test-suite-quality.md` and
`audits/2026-07-21_audit-round2/08-test-coverage-gaps.md` first; already-dispositioned decisions
(community-health files, the test-layout / `_extra` / four-benchmark-surfaces conventions now
documented in TESTING.md) are not re-raised, and items verified fixed since 2026-07-21 are noted as
fixed rather than re-listed.

Ran, on this box (`D:/ProgramData/anaconda3/python.exe -m pytest --no-cov -q -p no:randomly`):
`tests/test_parallel.py tests/test_distributed.py tests/test_monitoring.py` (40 passed, 7.05s) and a
`-k` selection over `test_parallel.py test_pandaslib_extra.py test_system_extra3.py
test_pythonlib_extra2.py` (7 passed) to confirm the vacuous tests cited below do pass today.
Also ran `--collect-only` over the whole suite (3539 tests collected, 18.2s), and executed
`pyutilz.dev.code_audit`'s own `scan_vacuous_assertions` / `scan_except_skip_masks_call_under_test`
against `tests/` directly. Static analysis: AST scans for assertion-free tests (43), tests whose
every assertion sits under an `if` guard (34), duplicate test-function names per class/module (0),
never-requested fixtures, and public-function-vs-test-corpus coverage (483 public functions,
26 never mentioned anywhere under `tests/`).

Verified fixed since 2026-07-21 (not re-raised): the `test_serialization.py` tautology,
`TestHeartbeatSql` in `test_distributed.py` (rewritten against the real `get_heartbeat_sql`),
`benchmark_pandaslib.py` (moved to `_benchmarks/bench_pandaslib.py`), `test_web.py:22-52` and
`:69-78`, the `test_numbalib.py` seed tests (now assert reproducibility through an `@njit` draw),
the `integration` marker (removed from `[tool.pytest.ini_options]`), the `gpu` marker (now enforced
via `pytest -m "not gpu"` in CI), and TESTING.md's new layout / `_extra` / benchmark-naming sections.

Findings: 3 High, 10 Medium, 8 Low (21 total).

## Findings

### F01. [High] The repo's own test-quality scanners are never run against `tests/` — 9 unfixed P1 findings they already detect — tests/test_meta/test_code_audit_baseline.py:38
- **Disposition**: OPEN
- **Category**: unenforced-self-tooling
- **Problem**: `pyutilz.dev.code_audit` ships test-quality scanners (`scan_vacuous_assertions`, `scan_except_skip_masks_call_under_test`, `scan_unenforced_docstring_invariants`) with full unit coverage in `tests/test_code_audit.py` (e.g. `test_except_skip_masks_real_call_flagged` at line 4337, `test_vacuous_assertion_bare_true_flagged` at line 2514). The self-scan meta-test passes `root=PYUTILZ_DIR` (`tests/test_meta/test_code_audit_baseline.py:38`), i.e. `src/pyutilz` only — `tests/` is never scanned. Running the scanners on `tests/` by hand returns **9 P1 `except_skip_masks_call_under_test` findings**, none baselined or fixed: `test_llm_factory.py:69`, `test_llm_live.py:182`, `test_llm_live.py:205`, `test_meta/test_module_alias_integrity.py:183`, `test_pandaslib_extra2.py:246`, `:256`, `:266`, `:276`, `test_smoke_untested_modules.py:50`.
- **Failure scenario**: pyutilz is the reference implementation every downstream consumer copies its code-audit wiring from, yet its own test suite is exempt. A genuine API break in `get_llm_provider("claude-code")` (test_llm_factory.py:67-72) or in `ensure_dataframe_float32_convertability`'s pyarrow branch (test_pandaslib_extra2.py:246-282) is silently reclassified as a skip; CI stays green with the tests never executing their assertions.
- **Suggested fix**: add a second baseline-driven meta-test (or a second `assert_no_new_code_audit_findings` call) with `root=TESTS_DIR` and its own `_code_audit_tests_baseline.json`, then drain the 9 P1s by narrowing each `try:` body to the bare import.

### F02. [High] `inspect.getsource()` string matching is still the primary regression guard in four files — still open since 2026-07-21 — tests/test_system.py:22, tests/test_parallel.py:48, tests/test_monitoring.py:21, tests/test_distributed.py:87
- **Disposition**: OPEN
- **Category**: source-inspection-instead-of-behavioral
- **Problem**: Raised as Medium on 2026-07-21; unfixed. Live sites: `tests/test_system.py:22` (`"subprocess.PIPE" in source`), `:120` (`"platform.system()" in source or "try:" in source`), `:156` (`"cpu_percent" in source`), `:351` (`"mi.private" in src`); `tests/test_parallel.py:48` (`"atexit.register" in source`), `:155` (`"select_device(3)" not in source`); `tests/test_monitoring.py:21` (`"_TIMEOUT_EXECUTOR" in source`); `tests/test_distributed.py:87`, `:108`, `:150`. The `test_distributed.py` ones additionally import via the `pyutilz.distributed` alias, which resolves fine (`src/pyutilz/system/distributed.py`), so their `pytest.skip("distributed module not available")` guards never fire and the source scan really does run. The repo's own conventions prefer behavioural tests over `getsource()` string assertions.
- **Failure scenario**: renaming `_TIMEOUT_EXECUTOR` to `_EXECUTOR_POOL`, or moving `atexit.register` into an imported helper, fails the test without any behaviour changing; conversely, keeping the string while breaking the behaviour (e.g. registering the atexit handler but never populating `_TEMP_DIRS`) passes. `test_system.py:112-127` is doubly weak — the assertion `"platform.system()" in source or "try:" in source` is satisfied by any file containing a `try:` anywhere.
- **Suggested fix**: replace each with a behavioural equivalent — assert one executor object is shared across two decorated calls (or count `ThreadPoolExecutor` constructions via `patch`); assert the `atexit`-registered callable actually removes a real temp dir; assert `get_own_memory_usage()` against a `patch`ed `psutil.Process.memory_info` returning distinct `rss` vs `private`.

### F03. [High] Seven hardware-detection tests pass unconditionally when the function under test returns falsy — tests/system/test_hardware_detection.py:40
- **Disposition**: OPEN
- **Category**: conditional-assertion
- **Problem**: `test_get_cpu_info` (line 40), `test_get_wmi_cpuinfo` (55), `test_get_lscpu_info` (69), `test_get_nvidia_smi_info` (113), `test_get_cuda_gpu_details` (127), `test_get_power_plan` (193), `test_get_battery_info` (204) all follow `result = f(); if result: assert ... else: print("[WARN] ...")`. Every assertion in each of these functions sits inside the truthy branch (verified by AST: no assertion outside an `if` in any of them).
- **Failure scenario**: a regression making `get_cpu_info()` return `{}` or `None` — an exception swallowed inside the probe, or a filter that strips every key — takes the `else` branch, prints `[WARN] py-cpuinfo not available`, and the test passes green on every platform. These are exactly the functions where "returned nothing" *is* the bug.
- **Suggested fix**: split into an availability precondition (`pytest.importorskip` / `pytest.skip` when the underlying tool is genuinely absent) and an unconditional assertion on the returned structure. For `get_battery_info` / `get_power_plan`, where a desktop legitimately has no data, assert `result is None or isinstance(result, dict)` plus the key contract when non-None, instead of printing a warning.

### F04. [Medium] `test_gpu_selection_not_hardcoded` has an if/else where both branches are `pass` — tests/test_parallel.py:146
- **Disposition**: OPEN
- **Category**: vacuous-assertion
- **Problem**: Lines 155-169: `if "cuda.select_device" in source:` guards the only assertion; inside it, `if "CUDA_VISIBLE_DEVICES" in source or "getenv" in source: pass else: pass`. Verified passing today (1 passed).
- **Failure scenario**: if `pyutilz.parallel` stops calling `cuda.select_device` (or the call moves into a helper module), the outer guard is false and the whole test becomes a no-op — including the `select_device(3)` hard-coding check it exists for.
- **Suggested fix**: delete the dead `if/else`, and make the outer condition an assertion of intent — behaviourally, monkeypatch `CUDA_VISIBLE_DEVICES` and assert the selected device index follows configuration.

### F05. [Medium] `report_actual_duration=True` test accepts the feature being switched off — tests/test_monitoring.py:96
- **Disposition**: OPEN
- **Category**: conditional-assertion
- **Problem**: `test_timeout_wrapper_report_duration` (lines 96-114) calls a function wrapped with `report_actual_duration=True`, then `if isinstance(result, tuple): assert actual_result == "done" and duration >= 0.1; else: assert result == "done"`. The `else` branch is exactly the behaviour the flag is supposed to change.
- **Failure scenario**: a regression where `report_actual_duration` is ignored (bare result returned) takes the `else` branch and passes. The one contract the test's docstring names is unverifiable.
- **Suggested fix**: `assert isinstance(result, tuple)` unconditionally, then unpack and assert both fields.

### F06. [Medium] `TestReportLargeObjects` — four tests, zero assertions — tests/test_system_extra3.py:692
- **Disposition**: OPEN
- **Category**: assertion-free-coverage-chasing
- **Problem**: `test_no_big_objects` (693), `test_with_big_objects` (699), `test_with_memory_snapshot` (705), `test_with_memory_snapshot_exception` (715) each patch `pyutilz.system.system.misc.asizeof`, call `report_large_objects(...)`, and end. No assertion, no `caplog`, no mock-call check. The class comment `# ── report_large_objects (lines 1659-1678) ──` states the intent plainly: line coverage.
- **Failure scenario**: `report_large_objects` reporting *nothing* (inverted `>=` on the `min_size_mb` threshold, or a swallowed exception in the reporting loop) passes all four. `test_no_big_objects` and `test_with_big_objects` differ only in the mocked size and produce identical unchecked outcomes, so they cannot even distinguish the threshold.
- **Suggested fix**: assert on `caplog` records (or a patched logger/print) — the 300 MB case must name the object, the 100-byte case must report nothing; the exception case must not propagate *and* must still emit the size report.

### F07. [Medium] `ensure_installed` tests assert nothing, including the one thing they name — tests/test_pythonlib_extra2.py:18
- **Disposition**: OPEN
- **Category**: assertion-free-coverage-chasing
- **Problem**: `test_ensure_installed_single_string` (18-21) patches `find_spec` to return `True` and comments `# should not install`, but never inspects `subprocess.check_call`. `test_ensure_installed_missing_package` (24-28) patches `check_call` with `side_effect=Exception("mock")` and asserts nothing about which package was requested. `test_ensure_installed_none` (31-32) asserts nothing.
- **Failure scenario**: a regression that ignores `find_spec` and pip-installs on every call passes `test_ensure_installed_single_string` — the wasted subprocess per import stays invisible. A regression passing the wrong package name to pip passes `test_ensure_installed_missing_package`.
- **Suggested fix**: patch `subprocess.check_call` with a `MagicMock` in both; assert `mock.assert_not_called()` in the present case and that `mock.call_args` contains `"nonexistent_pkg_xyz"` in the missing case.

### F08. [Medium] CI coverage gate is 20 points below actual coverage — .github/workflows/ci.yml:77
- **Disposition**: OPEN
- **Category**: pytest-config
- **Problem**: CI runs `pytest -m "not gpu" --cov=src/pyutilz ... --cov-fail-under=60`. `TESTING.md` line 3 documents current coverage as **79.6%**.
- **Failure scenario**: a change removing or bypassing ~19 percentage points of covered code — a whole subpackage's tests going silently skipped, cf. F01 and F10 — passes the gate. The gate cannot detect any realistic coverage regression.
- **Suggested fix**: raise `--cov-fail-under` to just under the measured value (e.g. 78) and treat it as a ratchet, refreshed upward the way the meta-test baselines already are.

### F09. [Medium] Two test files cover `pyutilz.stats.normality`; the newer one's docstring falsely claims the module was untested, and it violates TESTING.md's own layout rule — tests/test_normality.py:1
- **Disposition**: OPEN
- **Category**: test-organization
- **Problem**: `tests/stats/test_normality.py` (11 tests) and `tests/test_normality.py` (22 tests) both target `pyutilz.stats.normality`. The flat-root file opens: *"Behavioral coverage for pyutilz.stats.normality (previously untested -- no test_normality.py existed at all)"* — untrue; `tests/stats/test_normality.py` exists and is the calibration suite the 2026-07-21 audit singled out as good engineering. TESTING.md's "Test layout" section states explicitly that new tests for `stats/*` go in `tests/stats/`. These are the only duplicate test-file basenames in the repo (survivable only because both directories carry `__init__.py`).
- **Failure scenario**: a contributor checking whether `normality_verdict`'s degenerate branch is covered reads one file, sees a gap, and duplicates work — which is what already happened. `pytest tests/test_normality.py` runs half the coverage while looking complete.
- **Suggested fix**: merge the flat-root file into `tests/stats/` (e.g. `tests/stats/test_normality_kernels.py`) and delete the false docstring claim.

### F10. [Medium] Permanently-skipped test with an unnamed underlying reason — tests/test_logginglib.py:96
- **Disposition**: OPEN
- **Category**: skip-hiding-gap
- **Problem**: `@pytest.mark.skip(reason="Requires inflect module")` on `test_log_loaded_rows`; the body is `pass` with a comment repeating the reason. It is an unconditional `skip`, not `skipif` — it never runs even where `inflect` *is* installed, and `inflect` is declared in no optional-dependency group in `pyproject.toml`. `log_loaded_rows` therefore has zero coverage on every platform, permanently.
- **Failure scenario**: any bug in `log_loaded_rows` (pluralisation, row-count formatting, the log-dict write) ships unnoticed, and the skip presents as an environment limitation rather than "this function is untested".
- **Suggested fix**: write the test with `inflect = pytest.importorskip("inflect")` and add `inflect` to the `dev` extra so it actually runs in CI; or delete the placeholder and record `log_loaded_rows` as a known-untested public function.

### F11. [Medium] `except Exception: pytest.skip(...)` around the call under test — still open since 2026-07-21 — tests/test_llm_factory.py:67
- **Disposition**: OPEN
- **Category**: skip-masks-failure
- **Problem**: Raised on 2026-07-21 as "inconsistent exception-swallowing between near-duplicate tests"; unfixed. `test_cache_returns_same_instance` (67-73) wraps two real `get_llm_provider("claude-code")` calls in `try/except Exception: pytest.skip("claude-code provider not available in test env")`. The sibling `test_claude_code_variants_accepted` (59-65) does the opposite and `pytest.fail`s. Independently detected by the repo's own scanner (F01).
- **Failure scenario**: the caching contract — the entire point of the test — goes unverified whenever the provider constructor raises for any reason, including a genuine regression such as `_provider_cache` keying on an unhashable value.
- **Suggested fix**: narrow the guard to the provider import (or `pytest.importorskip`) and let any construction failure fail the test.

### F12. [Medium] No `filterwarnings` in the pytest config — pyproject.toml:350
- **Disposition**: OPEN
- **Category**: pytest-config
- **Problem**: `[tool.pytest.ini_options]` (lines 350-360) sets `testpaths`, `python_files/classes/functions`, `addopts = "-v --tb=short --strict-markers -ra --timeout=120"` and three markers. There is no `filterwarnings` key anywhere in `pyproject.toml` or any test file. A `--collect-only` run already emits third-party `DeprecationWarning`s (e.g. `weasel/util/config.py:8`, Click 9.0 removal) into the noise.
- **Failure scenario**: pyutilz's own `DeprecationWarning`s — the ones the PascalCase→snake_case deprecation tests in `test_db_extra.py` and `tests/test_meta/test_deprecation_warnings_present.py` exist to police — are indistinguishable from third-party noise, and a pandas/numpy `FutureWarning` announcing a behaviour change pyutilz relies on scrolls past unnoticed until the dependency actually breaks.
- **Suggested fix**: add `filterwarnings = ["error::DeprecationWarning:pyutilz.*", "ignore::DeprecationWarning"]` (or the ratchet form, `error` plus an explicit `ignore` list for known third-party emitters), matching the `--strict-markers` discipline already in place.

### F13. [Medium] Wall-clock threshold assertions a loaded box or CI runner can fail — tests/test_numbalib.py:131, tests/test_pythonlib.py:546
- **Disposition**: OPEN
- **Category**: flaky-timing
- **Problem**: `test_uses_join_not_concatenation` (test_numbalib.py:118-132) asserts `elapsed < 0.1` for `arr2str(list(range(1000)))` — enormous margin relative to the work, tiny in absolute terms on a shared CI runner or a paging Windows box. `tests/test_pythonlib.py:546` and `:558` assert `elapsed < 0.5` around `imitate_delay`; `tests/test_tokenizers_extra.py:118` asserts `< 1.0`. (`tests/performance/kernel_tuning/test_cache_concurrency.py:267` and `tests/test_similarity_coverage_gate_batch_kernel.py:113` use 5.0s budgets — comfortable, noted as acceptable.)
- **Failure scenario**: intermittent red CI unrelated to any code change, training contributors to re-run rather than investigate — the failure mode `tests/test_monitoring_extra.py:245-270` documents as having already happened here (2026-07-09 intermittent timeout failures).
- **Suggested fix**: for `arr2str`, assert the complexity property rather than a constant — time N and 10N and assert a sub-quadratic ratio — or raise the budget to something no non-quadratic implementation can exceed. For `imitate_delay`, the meaningful assertion is the lower bound, already present at line 557.

### F14. [Low] `pytest-randomly` is neither installed nor declared, yet three files' designs are justified by it — and conftest.py still monkeypatches thinc globally for it — tests/conftest.py:22
- **Disposition**: OPEN
- **Category**: dead-test-infrastructure
- **Problem**: `pytest-randomly` appears in none of `pyproject.toml`'s dev dependencies (lines 305-311: pytest, pytest-cov, pytest-benchmark, pytest-asyncio, pytest-instafail, pytest-progress, pytest-timeout) and is absent from the plugin banner of a live run on this box. `tests/conftest.py:9-53` is a ~45-line compat shim for it that nonetheless *unconditionally* rebinds `thinc.util.fix_random_seed` at conftest import time (line 42), for every session, plugin present or not. `tests/test_image.py:16-29` and `tests/test_pandaslib_extra.py:154` both justify their designs with "order randomized by pytest-randomly".
- **Failure scenario**: the suite always runs in deterministic file order, so genuine order-dependence (F15; the `_provider_cache` sharing between `test_llm_factory.py` and `test_llm_providers.py`) is never exercised — while the shim mutates a third-party library's global RNG entry point on every run, a side effect nothing in the declared environment needs.
- **Suggested fix**: decide one way — add `pytest-randomly` to the `dev` extra and keep the shim (recovering the order-independence checking the comments assume), or drop the shim and the comments referencing it.

### F15. [Low] Module-scoped `sys.modules` surgery leaves the parent package attribute bound to the stub-built module — tests/test_image.py:42
- **Disposition**: OPEN
- **Category**: module-reload-hack
- **Problem**: The `_stub_pil` fixture pops `pyutilz.core.image` from `sys.modules` (line 42), installs `MagicMock` PIL stubs, and on teardown restores `sys.modules` (45-53). It does not restore `pyutilz.core.image` as an *attribute* of the `pyutilz.core` package. Verified directly: after `sys.modules.pop("pyutilz.core.image")`, `getattr(pyutilz.core, "image")` still resolves to the old module object — so the re-import inside the fixture rebinds that attribute to the stub-built module, and teardown never undoes it. The repo bans `importlib.reload` for exactly this identity-splitting reason (`tests/test_meta/test_no_module_reload.py`), but the ban does not cover `sys.modules.pop` + re-import.
- **Failure scenario**: latent today — nothing else accesses `pyutilz.core.image` by attribute (only the lazy alias at `src/pyutilz/__init__.py:16`, which goes through `sys.modules` and is correctly restored). As soon as any test does `from pyutilz.core import image`, it silently gets the MagicMock-PIL-bound module, and its `IFDRational`/`isinstance` behaviour depends on file order.
- **Suggested fix**: restore the parent attribute in teardown (`setattr(pyutilz.core, "image", saved_image_mod)` / `delattr` when there was none), and extend `test_no_module_reload.py`'s AST check to flag `sys.modules.pop`-then-re-import outside a reviewed whitelist.

### F16. [Low] Four conftest fixtures are dead — tests/conftest.py:210
- **Disposition**: OPEN
- **Category**: fixture-misuse
- **Problem**: `mixed_types_df` (210), `temp_dir` (222), `float_with_integers_df` (229) and `constant_columns_df` (235) are requested by no test anywhere under `tests/` (grep across every test file; only `sample_df` at line 204 is used, by `test_pandaslib.py`). `temp_dir` also duplicates pytest's builtin `tmp_path`.
- **Failure scenario**: no runtime effect, but they read as "these DataFrame shapes are covered somewhere" when nothing exercises them, and `temp_dir` invites new tests to use a hand-rolled fixture instead of `tmp_path`.
- **Suggested fix**: delete the four, or move the three DataFrame fixtures into `tests/test_pandaslib*.py` where the shapes are relevant, and use `tmp_path` in place of `temp_dir`.

### F17. [Low] The `slow` marker's documented deselection is never used anywhere — pyproject.toml:357
- **Disposition**: OPEN
- **Category**: pytest-config
- **Problem**: `"slow: marks tests as slow (deselect with '-m \"not slow\"')"` is declared, but exactly **one** test carries it (`tests/system/test_hardware_detection.py:352`, a 2-second sleep), and no workflow, pre-commit hook, or TESTING.md command ever passes `-m "not slow"`. The genuinely slow tests are unmarked: `tests/test_monitoring_extra.py:254` (15 × 0.2s timeouts plus a 30s-sleeping worker), `tests/test_code_audit.py:2782`/`:2798`/`:2824` (1s sleeps), `tests/test_monitoring.py:44`, and `test_monitoring.py:173`'s `(1, 2, True)` parametrisation. Partially open since 2026-07-21 — `gpu` is now enforced and `integration` was removed, but `slow` was not addressed.
- **Failure scenario**: the marker's promise of a fast inner-loop subset does not hold — `-m "not slow"` deselects one 2s test out of 3539 — so contributors run everything or nothing.
- **Suggested fix**: mark the real offenders (CI already emits `--durations=200`; use it to pick them) and add a documented `pytest -m "not slow"` recipe to TESTING.md, or drop the marker and its deselect advice.

### F18. [Low] Stale comment plus a weakened assertion hides the actual classification — tests/test_pandaslib_extra.py:336
- **Disposition**: OPEN
- **Category**: weak-assertion
- **Problem**: `test_dtype_param_string` guards on `if "object" in dtype.name:` and otherwise asserts only `is_bool is False`, with the comment *"StringDtype: classified as numeric by default (known limitation)"*. Verified on this box: `pd.Series(["x"]).dtype.name == "str"` (so the `else` branch is taken) and `classify_column_types(dtype=...)` returns `(False, True, False, False, False)` — `is_obj is True`, `is_num is False`. The comment is wrong, and the branch that actually runs asserts a fact true of every non-bool dtype.
- **Failure scenario**: string columns being reclassified as numeric — the exact "known limitation" the comment describes, and the one that would silently feed strings into numeric pipelines — still passes this test.
- **Suggested fix**: assert `is_obj is True and is_num is False` unconditionally and delete the stale comment (the limitation it describes does not reproduce).

### F19. [Low] Either-shape assertions accept both possible contracts — tests/test_logginglib.py:40
- **Disposition**: OPEN
- **Category**: weak-assertion
- **Problem**: `test_log_result` (around line 40) asserts `"results" in log or "test_key" in log`, then `if "results" in log: assert log["results"]["test_key"] == 42 else: assert log["test_key"] == 42`. `test_log_results` (46) and `test_log_result_with_none` (152) repeat the pattern.
- **Failure scenario**: `log_result` moving values between the nested `results` dict and the top level — a breaking change for every consumer reading `log["results"]` — passes unchanged. The tests pin no contract at all for where results live.
- **Suggested fix**: read `initialize_function_log`'s actual shape once and assert the nested form unconditionally (a single `assert log["results"]["test_key"] == 42`).

### F20. [Low] Public functions with zero mention anywhere in `tests/` — src/pyutilz/core/pythonlib.py:694
- **Disposition**: OPEN
- **Category**: untested-public-api
- **Problem**: Of 483 module-level public functions under `src/pyutilz/`, 26 are never named in any file under `tests/`. Excluding the coverage-omitted modules (`cloud/`, `web/browser.py`, `dev/dashlib.py`, `dev/notebook_init.py`, `text/tokenizers.py`, `system/scheduling/*` — all listed in `[tool.coverage.run] omit`), these remain untested and non-exempt: `load_file` (`core/pythonlib.py:694` — also referenced by nothing in `src/`, i.e. dead public API), `div0` (`data/numpylib.py:75`, the zero-division helper `data/numpylib.py:100` builds on), `is_llm_refusal` (`llm/base.py:205`, used at `llm/base.py:411`), `longest_prefix_lookup` (`llm/base.py:166`, used by `llm/anthropic_provider.py:129` for max-output-token resolution), `tokenize_text` (`text/strings/textentropy.py:23`, re-exported at `text/strings/__init__.py:141`), `spacy_sent_tokenize` (`text/strings/webtext.py:20`), `is_local_path` (`database/deltalakes.py:21`, used at `deltalakes.py:72`), plus six `dev/` tooling entry points that are themselves in `__all__`: `check_all` (`dev/code_audit/field_text_agreement.py:280`), `normalise_text` (same file, line 47), `get_scanners` (`dev/code_audit/registry.py:190`), `findings_ratchet` (`dev/meta_test_utils.py:599`), `snake_case_variants_of` (`dev/meta_test_utils.py:442`), `unbacked_audit_dispositions` (`dev/meta_test_utils.py:623`).
- **Failure scenario**: `longest_prefix_lookup` returning the wrong prefix silently caps Anthropic output tokens at the 64000 fallback; a regression in `div0`'s `na_fill` handling corrupts every consumer of `data/numpylib.py:100` — neither has a test to catch it. `findings_ratchet` and `get_scanners` are part of the shared meta-test harness pyutilz exports to six downstream repos, untested here.
- **Suggested fix**: add behavioural tests for the seven runtime functions (`div0` zero/NaN/inf fills, `is_llm_refusal` positive and negative phrasings, `longest_prefix_lookup` exact/prefix/fallback, `load_file`, `tokenize_text`, `is_local_path` local-vs-remote); for the `dev/` helpers, either test them or record an exemption with a cited consumer alongside `test_test_source_parity.py`'s `_TEST_EXEMPT_MODULES`. `load_file` has no caller anywhere — decide test-or-delete.

### F21. [Low] TESTING.md's headline numbers are stale and its documented coverage command fails on Windows — TESTING.md:3
- **Disposition**: OPEN
- **Category**: docs
- **Problem**: TESTING.md line 3 says "1900+ tests ... ~3 min"; `--collect-only` today collects **3539** tests. The "Running tests" block documents `pytest --cov=src/pyutilz --cov-report=term-missing` with no mention that pytest-cov raises `PermissionError` on Windows (the reason local runs in this repo need `--no-cov`), even though the file already carries a dedicated "Local pip-audit on Windows" section for the analogous cp1251 problem.
- **Failure scenario**: a Windows contributor follows the documented coverage command, hits an opaque `PermissionError`, and finds nothing in TESTING.md pointing at `--no-cov`.
- **Suggested fix**: refresh the count and runtime, and add a one-line Windows note next to the `--cov` recipe (use `--no-cov` locally on Windows; coverage is measured in CI on ubuntu-latest).
