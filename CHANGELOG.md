# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-05-04

### Added — OpenRouter unified ``reasoning`` field + widened ``thinking=`` parameter (mixed bool|str)

The base OpenAI-compatible provider's ``generate(thinking=...)`` /
``generate_stream(thinking=...)`` parameter is widened from ``bool | None``
to ``bool | str | None``. New string values: ``"low"``, ``"medium"``,
``"high"``, ``"minimal"`` map to the upstream's effort-tier where the
upstream supports one (OpenRouter's unified ``reasoning.effort`` field;
OpenAI ``reasoning_effort``). ``True`` is mapped to ``"medium"`` for
effort-string upstreams; non-empty string is coerced to ``True`` for
bool-flag upstreams (DeepSeek V4 ``thinking.type``). Empty string ``""``
is treated as ``False`` defensively.

New static helper ``OpenAICompatibleProvider._normalize_thinking(value)
-> (enabled, effort)`` lets each provider's ``_thinking_request_field``
override pick the half of the contract its API requires.

**OpenRouter override (new)**:
``OpenRouterProvider._thinking_request_field(thinking)`` emits OR's
unified ``reasoning`` field that auto-routes upstream:
- ``False`` / ``""`` -> ``{"reasoning": {"exclude": True}}``
- ``True`` -> ``{"reasoning": {"effort": "medium"}}``
- ``"low"`` / ``"medium"`` / ``"high"`` / ``"minimal"`` (or any other
  str) -> ``{"reasoning": {"effort": <str>}}`` (passed through; unknown
  strings forwarded for OR's edge to accept or 400)

Pre-fix the OR provider inherited the base class default (return
``None``) so any ``thinking=...`` argument was silently ignored on OR.
Now it routes via OR's ``reasoning`` field which OR auto-translates to
the correct upstream-specific shape (Anthropic ``thinking``, OpenAI
``reasoning_effort``, DeepSeek V4 thinking-toggle, etc.) based on the
resolved model ID. See https://openrouter.ai/docs/use-cases/reasoning-tokens.

**DeepSeek override updated**:
``DeepSeekProvider._thinking_request_field`` now goes through
``_normalize_thinking`` so callers passing ``thinking="high"`` (effort
string) still get DeepSeek's bool-flag (``{"thinking": {"type":
"enabled"}}``). Empty string disables.

14 new tests: 9 in ``tests/test_llm_openrouter.py``
(``TestThinkingRequestField``) + 5 in ``tests/test_llm_deepseek.py``
covering the bool-flag string-coercion path. 176 LLM regression tests
green.

## [Unreleased] — 2026-05-02

### Added — Live model-health pre-flight & enriched ``list_openrouter_models``

Two new ``OpenRouterProvider`` methods backed by ``GET /api/v1/models/{slug}/endpoints``
(auth required, but **not billed** against credits — i.e. a "free ping"):

- ``check_model_health(model=None)`` — per-upstream ``status``,
  ``uptime_5m`` / ``30m`` / ``1d``, ``latency_p50_ms`` / ``p95``,
  ``throughput_p50_tps``, ``context_length``, ``max_completion_tokens``,
  ``pricing``, ``supported_parameters``, ``quantization``,
  ``supports_implicit_caching``. Plus best-of aggregates across all
  upstreams: ``best_uptime_30m``, ``best_latency_p50_ms``,
  ``best_throughput_p50_tps``.
- ``is_model_healthy(model=None, min_uptime=0.99)`` — boolean guard;
  returns ``False`` (rather than propagating) on network errors so a
  failing pre-flight just defers the batch instead of crashing it.

``list_openrouter_models`` becomes a two-stage pipeline:

1. **Stage 1 (offline)** — same /api/v1/models filter as before
   (``name_contains``, ``max_input_per_1m``).
2. **Stage 2 (live, opt-in via ``return_only_healthy=True``)** — concurrent
   ``ThreadPoolExecutor`` fetches /endpoints for each Stage-1 survivor,
   drops anyone whose ``best_uptime_30m`` < ``min_uptime`` (default 0.95),
   and attaches the full health block to each surviving row.

New params: ``return_only_healthy=True`` (default), ``min_uptime=0.95``,
``api_key`` (overrides env), ``max_workers=16``, ``timeout=30.0``.
``sort_by`` extended with ``"uptime"`` / ``"latency"`` / ``"throughput"``
(use the best-of aggregate per row).

**Backward compat**: existing callers keep working. With NO API key
configured, Stage 2 logs a warning and degrades to Stage-1-only output —
matches the old offline behaviour. Tests using the old shape pass
``return_only_healthy=False`` to opt out cleanly.

23 new unit tests in this slice; 286 regression tests on touched modules
green (full LLM suite + provider meta-tests).

### Added — Full OpenRouter usage / response capture surface

`OpenRouterProvider` now records every field OR exposes per request:

- **`usage.cost_details.upstream_inference_cost`** → `last_/total_upstream_inference_cost_usd` (BYOK only — bare upstream price separated from any OR markup)
- **`usage.prompt_tokens_details.audio_tokens`** → `last_/total_audio_tokens` (audio-modal models)
- **Response-level metadata** → `last_generation_id` (for /generation lookup), `last_upstream_provider` (which backend actually served — critical when debugging routing), `last_upstream_model` (resolved model when `models_fallback` triggered), `last_native_finish_reason` (upstream's raw finish code: `tool_calls`, `end_turn`, `content_filter`, …)

New methods:

- **`fetch_generation_stats(generation_id=None)`** — hits `GET /api/v1/generation?id=…` for the post-hoc audit shape (~30 fields incl. `latency`, `generation_time`, `moderation_latency`, `provider_responses` per-attempt log, `cache_discount`, `response_cache_source_id`, `is_byok`). Defaults to `self.last_generation_id`. Use after a stream where the inline usage chunk was dropped, or for retroactive cost reconciliation.
- **`last_call_summary()`** — one-shot dict of every `last_*` metric (cost, tokens, cache, audio, generation id, upstream provider/model, finish reasons). Convenience for ad-hoc inspection / structured logging.

Extension to `get_session_cost()`: now also includes `upstream_inference_cost_usd`, `last_upstream_inference_cost_usd`, `audio_tokens`.

Tiny base-class addition: `_track_provider_specific_response(data)` hook on `OpenAICompatibleProvider`, mirror of the existing `_track_provider_specific_usage` but for response-level fields outside the `usage` block. Default no-op; OR uses it.

22 new unit tests; 264 regression tests on touched modules all green.

### Added — OpenRouter provider (meta-provider, 200+ models behind one API)

New `pyutilz.llm.openrouter_provider.OpenRouterProvider`, factory keys
`"openrouter"` / `"or"` / `"router"`, env var `OPENROUTER_API_KEY`.
Inherits from `OpenAICompatibleProvider` (OpenRouter is OpenAI-compatible
at the wire level), with three meta-provider-specific touches:

- **Authoritative cost via `usage.cost`** — every OR response carries the
  USD billed by the upstream provider. We track it as ground truth in
  `total_actual_cost_usd` / `last_actual_cost_usd`, surfaced under
  `get_session_cost()["actual_cost_usd"]`. Per-token estimates from
  the model catalogue would be wrong the moment OR reroutes to a
  different backend; the upstream-reported field is the only honest
  source. Captured via a tiny new `_track_provider_specific_usage(usage)`
  hook on the OpenAI-compat base class (no-op for other providers).
- **Lazy `/api/v1/models` catalogue** for `estimate_cost()` predictions —
  fetched once per process, cached process-wide; degrades to zero on
  network/parse failure (estimate isn't load-bearing). Public helper
  `pyutilz.llm.list_openrouter_models(name_contains=…, sort_by=…,
  max_input_per_1m=…, refresh=False)` returns the full catalogue so
  callers can browse pricing / context windows / supported parameters.
- **Routing knobs as hashable kwargs** so the factory's
  `tuple(sorted(kwargs.items()))` cache key keeps working:
  `provider_order`, `provider_ignore`, `provider_sort` (`"price"` /
  `"throughput"` / `"latency"`), `provider_allow_fallbacks`,
  `models_fallback`. `app_name` / `site_url` set `X-Title` and
  `HTTP-Referer` for the public openrouter.ai/rankings dashboard.

- **Per-model limits respect upstream caps** — `context_window` and
  `max_output_tokens` resolve from `top_provider.context_length` /
  `top_provider.max_completion_tokens` in the catalogue, falling back to
  the model-level `context_length` and class defaults. Matters because
  OR routes through one specific upstream that may cap shorter than the
  model's theoretical max (e.g. a 1M-context model served by a provider
  exposing only 200K). The upstream cap is what triggers `400: prompt
  too long`; trusting the theoretical max would hand callers a footgun.
- **Account introspection** — `check_account_limits()` calls
  `/api/v1/key` (returns `limit_remaining`, `usage_daily/weekly/monthly`,
  `byok_usage*`, `is_free_tier`, `rate_limit`); `get_account_credits()`
  calls `/api/v1/credits` for the simpler total-credits view. Both
  unwrap OR's `{"data": {...}}` envelope.

54 unit tests (`tests/test_llm_openrouter.py`); regression tests on
touched modules (base, deepseek, xai, factory, meta-tests) all green.

### Added — Unified account-credits / rate-limit interface across all LLM providers

Two new methods on `LLMProvider` ABC:
- `async get_account_credits() -> dict` — billing snapshot (normalized
  schema: `balance_usd`, `total_granted`, `total_used`, `currency`,
  `is_available`, plus provider's `raw` response)
- `async check_account_limits() -> dict` — quota / rate-limit snapshot

Implementation matrix (researched against current public docs, not
guessed):

| Provider     | get_account_credits     | check_account_limits |
|--------------|-------------------------|----------------------|
| OpenRouter   | ✓ /api/v1/credits       | ✓ /api/v1/key        |
| DeepSeek     | ✓ /user/balance         | NotImplementedError  |
| Anthropic    | NotImplementedError     | NotImplementedError  |
| OpenAI       | NotImplementedError     | NotImplementedError  |
| xAI          | NotImplementedError     | NotImplementedError  |
| Gemini       | NotImplementedError     | NotImplementedError  |
| Claude Code  | NotImplementedError     | NotImplementedError  |

NotImplementedError stubs (rather than the base default) carry
provider-specific guidance: where to look in the console (URL),
which response headers to inspect (`x-ratelimit-*`,
`anthropic-ratelimit-*`), or why the API can't expose this (e.g.
Gemini routes billing through GCP Cloud Billing API requiring
separate auth; Claude Code is a Max-subscription product without
per-token credits). Honest "not supported" beats guessed endpoints
that 404 in production.

176 regression tests green across base, factory, deepseek, xai,
openai_compat, providers, retry, and meta-test suites; +18 new
tests in `tests/test_llm_account_credits.py` covering the base
default, DeepSeek's real implementation, every stub's error
message, and the meta-test that every provider exposes both
methods (polymorphic `await provider.get_account_credits()`).

## [Unreleased] — 2026-04-28

### Added (later that day) — `pyutilz.dev.meta_test_utils` + 11 more meta-tests

`pyutilz/src/pyutilz/dev/meta_test_utils.py` — a reusable library of
helpers for package-level meta-tests, factored out so both pyutilz and
its dependent (mlframe) import the same plumbing rather than duplicating
~400 LOC. Public API: `consumer_corpus`, `enumerate_test_files`,
`public_top_level_symbols`, `strip_lineno`, `capture_signature`,
`capture_module_surface`, `scan_todo_markers`,
`count_user_deferred_entries`, `snake_case_variants_of`, `safe_import`.

11 new meta-tests in `tests/test_meta/`:

- `test_deferred_drift.py` (A2) — counts `_USER_DEFERRED_*` /
  `_GRANDFATHERED` whitelist entries across every meta-test, fails on
  growth vs `_debt_baseline.json`. Net counter visible per run; refresh
  via `--refresh-debt-baseline`. Baseline: 4 whitelists, 20 entries.
- `test_provider_contract.py` (D1) — every concrete LLM provider
  inherits from `LLMProvider`, overrides every abstract method, and its
  override signatures stay compatible with the base (no dropped
  required parameters).
- `test_encoding_consistency.py` (D2) — every builtin `open(...)` call
  in production code passes `encoding=` (or uses `"b"` mode). Critical
  for Windows cp1251/cp1252 where the default codec crashes on non-
  ASCII files. Defers 8 known offenders in `_USER_DEFERRED_CALLS` for
  bulk fix later.
- `test_provider_cache_concurrency.py` (D3) — 20 concurrent
  `get_llm_provider("anthropic", ...)` callers share the same instance,
  the constructor runs exactly once, and distinct kwargs produce
  distinct instances. Catches a future refactor breaking the
  double-checked-locking pattern in `_provider_cache`.
- `test_public_docstrings.py` (E1) — snapshot-based check for new
  public symbols without a docstring. Baseline: 176 undocumented.
  Refresh: `--refresh-docstring-baseline`.
- `test_public_annotations.py` (E2) — snapshot-based check for new
  public functions without complete type annotations (return + every
  non-self/cls param). Baseline: 157 unannotated. Refresh:
  `--refresh-annotation-baseline`.
- `test_version_consistency.py` (E3) — `pyutilz.__version__`,
  `pyutilz.version.__version__`, and `pyproject.toml::[project].version`
  must all agree. Both sources currently `1.0.0` — pass.
- `test_no_import_cycles.py` (E4) — Tarjan's SCC over the AST-built
  import graph; flags multi-node cycles. pyutilz currently has none
  (3 single-node `__init__.py` self-references are intentional re-export
  patterns; not flagged).
- `test_no_unicode_in_console_output.py` (E5) — snapshot-based check
  for non-ASCII string literals in `print(...)` / `logger.*(...)` calls.
  Baseline: 27 existing offenders. Critical for Windows stdout.
- `test_meta_meta.py` (F1+F2+F3) — actionable failure messages, no
  private-internals reach-ins (with whitelist citing
  `_create_lazy_module`, `_PROVIDER_MODULES`, `_provider_cache` as
  legitimate test surfaces), perf-budget overrides match real test
  names.

### Existing tests refactored to use `pyutilz.dev.meta_test_utils`

`test_dead_helpers.py`, `test_todo_hygiene.py`, `test_test_source_parity.py`,
`test_api_stability.py` now import shared building blocks instead of
re-implementing them inline.

### Total meta-test footprint after this batch: 18 files, 45 tests, ≈ 35 s wall-clock.

## [Unreleased] — 2026-04-28

### Added — meta-test suite (`tests/test_meta/`, 29 tests, ≈ 66 s wall-clock)

A package-level static-check suite that catches whole classes of drift
without exercising the runtime behaviour. Each test is independently
runnable, has its own per-test whitelist for "drain over time" debt,
and is wired into `.pre-commit-config.yaml` so misconfigurations get
caught at commit time instead of in downstream CI.

- **PT-1 (`test_provider_registration.py`)** — every canonical name in `llm.factory._PROVIDER_MODULES` points to an importable module / class, every alias in `_ALIASES` resolves to a real canonical entry, and aliases never collide with canonical names. Catches "added a new provider, forgot the registry entry" — surfaces the bug *before* the first user request crashes inside `importlib`.
- **PT-2 (`test_module_alias_integrity.py`)** — every value in `_MODULE_ALIASES` (the 24-entry backward-compat map under `pyutilz/__init__.py`) imports cleanly, the proxy module installed at `pyutilz.<alias>` resolves a real public symbol on attribute access, alias keys never collide with sub-package names, and every alias target lives under the `pyutilz.` namespace. **Surfaced + fixed a real bug:** `pyutilz/web/browser.py:41` was importing `from . import pythonlib` (i.e. `pyutilz.web.pythonlib`), but `pythonlib` lives under `pyutilz.core`. Fixed to `from pyutilz.core import pythonlib`.
- **PT-3 (`test_test_source_parity.py`)** — every non-trivial production module under `src/pyutilz/` has at least one corresponding `tests/test_<name>.py` (or aliased name); reverse direction also flags test files without a target module. Surfaced 8 modules without test coverage held in `_USER_DEFERRED_MODULES` for later attention.
- **PT-4 (`test_todo_hygiene.py`)** — every `TODO` / `FIXME` / `XXX` / `HACK` comment in production code carries an attribution (assignee in parens or ISO date or `@assignee`). Currently 0 bare markers — the codebase is clean.
- **PT-5 (`test_dead_helpers.py`)** — public `def` / `class` symbols inside `pyutilz/llm/` (the only sub-tree where helpers are unambiguously internal) must be referenced ≥ 2× in the production corpus. Surfaced `LLMTruncationError` and `parse_retry_after` — held in `_USER_DEFERRED_DEAD_HELPERS` pending review.
- **PT-6 (`test_api_stability.py`)** — captures the public surface (top-level `__all__` + alias map + every alias-target's public symbol set with signatures + class MROs) into `tests/test_meta/_api_snapshot.json`. Renames / removals fail the test; additions are silent. Refresh after intentional API changes via `pytest tests/test_meta/test_api_stability.py --refresh-api-snapshot`. Initial snapshot: 27 aliases, 649 symbols.
- **PT-7 (`test_lazy_import_safety.py`)** — exercises the `_create_lazy_module` proxy infrastructure under realistic access patterns: unknown dunder access raises `AttributeError` (so `hasattr(mod, '__weird__')` works), proxy returns the same object as direct import (no double-wrap), repeated accesses are consistent, alias dict survives `importlib.reload`.
- **PT-8 (`test_optional_deps_isolation.py`)** — `import pyutilz` succeeds with every optional-dep group masked (pandas / polars / database / web / cloud / nlp / llm). `pyutilz`, `pyutilz.core`, `pyutilz.text` import safely with **all** optional deps masked. Each scenario runs in a sub-process so the masking can't leak between tests.
- **PT-9 (`test_no_top_level_side_effects.py`)** — importing pyutilz and every probed sub-module performs zero network I/O. Sockets and `urllib.request.urlopen` are blocked in a sub-process; any module attempting a request at module-load time fails the test with the offending call captured.
- **`.pre-commit-config.yaml`** — runs the meta-test suite on every commit (≈ 30–60 s). A `manual`-stage variant skips PT-8 / PT-9 (the sub-process tests) and runs in ≈ 5 s for tight inner-loop work.

### Fixed

- **`pyutilz/web/browser.py`**: `from . import pythonlib` was unreachable (no such module under `pyutilz/web/`). Now imports `pythonlib` from `pyutilz.core`. Surfaced by PT-2 on its first run.

### Changed

- **`llm.deepseek_provider`**: updated model registry and pricing for the
  V4 launch (April 2026). Added `deepseek-v4-flash` (1M context, 384K max
  output, $0.14/$0.0028/$0.28 per 1M input-miss / input-hit / output) and
  `deepseek-v4-pro` (1M / 384K, $1.74/$0.0145/$3.48). Cache-hit rates now
  reflect the 2026-04-26 reduction to 1/10 of launch price. Default model
  switched from `deepseek-reasoner` to `deepseek-v4-flash`. Legacy aliases
  `deepseek-chat` / `deepseek-reasoner` retained (deprecated 2026-07-24).
- **`llm.xai_provider`**: refreshed Grok pricing (April 2026). Fixed prior
  1000x error on grok-4.20 family ($2000/$6000 → correct $2/$6 per 1M
  tokens). Added `grok-4.20-beta` (general flagship alias) and `grok-4`
  (alias for `grok-4-0709`). Added per-model context windows for grok-4
  (256K) and grok-3 family (131K). Confirmed grok-4.20 cache hit at
  $0.20/M (90% discount on input miss).

## [Unreleased] — 2026-04-21

### Added

- **`tqdmu_lazy_start(iterable, **kwargs)` in `system.system`**: drop-in
  for `tqdmu(iterable, **kwargs)` that starts the elapsed timer at the
  first iteration, not at bar construction. Prevents the stale-timer
  artefact (e.g. `desc: 0/N [6:27:44<?]`) that occurs when the caller
  does heavy work between building the iterable and pulling the first
  item. Underlying bar is still `tqdmu` — same environment-aware
  selection between ipython-notebook and terminal back-ends.
- **`deep: bool = True` kwarg on `data.pandaslib.get_df_memory_consumption`**:
  default preserves existing byte-precise behaviour.
  `get_df_memory_consumption(df, deep=False)` returns
  `df.memory_usage(deep=False).sum()` for pandas — O(cols), milliseconds,
  accounting for object columns at pointer size only. Use when the
  consumer is a coarse heuristic (e.g. GPU-RAM fit check) on frames
  with million-unique string columns where `deep=True` is pathological
  (O(rows * avg_str_len), minutes on multi-GB frames). Polars branch
  is unchanged (`.estimated_size()` is already O(cols)).

## [1.0.0] - 2026-02-18

### Added - Hardware Detection Migration
- **~30 hardware detection functions** migrated from ml_perf_test project to `system.system` module
- **CPU Detection**:
  - `get_cpu_info()` - Enhanced CPU detection via py-cpuinfo with better filtering
  - `get_wmi_cpuinfo()` - Windows WMI CPU detection with detailed hardware info
  - `get_lscpu_info()` - Linux lscpu parser with automatic type conversion
  - `get_linux_board_info()` - Linux motherboard info from /sys/devices
  - `parse_dmidecode_info()` - Linux dmidecode parser (BIOS, memory, etc.)
- **GPU Detection** (replaces old functions):
  - `get_nvidia_smi_info()` - Rich nvidia-smi XML parsing with full GPU stats
  - `get_gpu_cuda_capabilities()` - CUDA device attributes via numba.cuda
  - `get_cuda_gpu_details()` - Combined nvidia-smi + CUDA capabilities
  - `CUDA_SM_TO_CORES` - Compute capability to CUDA cores mapping constant
- **Power & Large Pages**:
  - `check_large_pages_support()` - Cross-platform large pages detection (Windows/Linux/macOS)
  - `get_power_plan()` - Cross-platform power plan detection
  - `get_battery_info()` - Battery status and charge level
- **WMI Helpers** (Windows):
  - `get_wmi_obj_as_dict()` - WMI object to dict conversion with type handling
  - `summarize_devices()` - Hardware aggregation with counts
  - `dict_to_tuple()` - Dictionary hashing helper
  - `decode_memory_type()` - DDR type decoder (DDR3/DDR4/DDR5)
  - `decode_cpu_upgrade_method()` - CPU socket type decoder
  - `summarize_system_info()` - Complete Windows system summary (GPU, RAM, Cache, BIOS)
- **OS & Software**:
  - `get_os_info()` - Enhanced OS detection with detailed info
  - `get_python_info()` - Python implementation and version detection
- **Monitoring**:
  - `ensure_idle_devices()` - Wait for CPU/GPU idle before benchmarks
  - `system.hardware_monitor.UtilizationMonitor` - Background thread monitoring for CPU/GPU/RAM utilization
- **Utilities**:
  - `remove_nas()` - Recursive N/A removal from dicts with type conversion

### Changed
- **`get_system_info()` enhanced** with new fields while maintaining backward compatibility:
  - `cpu_wmi_info`, `cpu_lscpu_info`, `cpu_board_info` (platform-specific)
  - `gpu_nvidia_smi_info`, `gpu_cuda_capabilities` (replaces old gpuinfo)
  - `large_pages_support`, `power_plan`, `battery_info`
  - `system_wmi_summary`, `dmidecode_info` (detailed hardware)
  - **Backward compatible**: Existing fields (`host_name`, `os_machine_guid`, `os_serial`) preserved for `distributed.py`

### Removed
- **Old GPU functions** (replaced with superior ml_perf_test implementations):
  - `compute_total_gpus_ram()` - Use `get_nvidia_smi_info()` instead
  - `get_gpuinfo_gpu_info()` - Use `get_nvidia_smi_info()` instead
  - `get_gpuutil_gpu_info()` - Use `get_nvidia_smi_info()` instead
  - `get_pycuda_gpu_info()` - Use `get_gpu_cuda_capabilities()` instead

### Dependencies
- **New optional dependencies** in `[system]` extra:
  - `py-cpuinfo>=9.0` - Enhanced CPU detection
  - `GPUtil>=1.4` - GPU monitoring
  - `xmltodict>=0.13` - nvidia-smi XML parsing
  - `pypiwin32>=223` - WMI support on Windows

### Testing
- **20 new tests** for hardware detection functions (`test_hardware_detection.py`)
- **95% success rate** on Windows (19/20 tests passing, 1 skipped for Linux)
- **Full test coverage** for CPU, GPU, power, OS, and monitoring functions
- **Cross-platform testing** with platform-specific markers

### Documentation
- **MIGRATION_TEST_REPORT.md** - Comprehensive migration and testing documentation
- Complete test results with hardware-specific outputs
- Dependency installation guide
- Integration examples for ml_perf_test project

### Migration Notes
- **ml_perf_test integration**: Hardware detection functions now imported from pyutilz
- **~1500 lines removed** from ml_perf_test.py (migrated to pyutilz)
- **Graceful dependency handling**: Functions return None with warnings if optional deps unavailable
- **Platform compatibility**: Proper guards for Windows/Linux/macOS-specific functionality

## [0.90] - 2026-02-18

### Added
- Public GitHub release with full packaging infrastructure
- Comprehensive test suite (142 tests passing)
- CI/CD automation with GitHub Actions
- Quality badges (CI, coverage, Codacy, security)
- Modern packaging with pyproject.toml
- Professional README with documentation
- CHANGELOG for version tracking
- CONTRIBUTING guidelines for developers
- Code coverage measurement with pytest-cov
- Security scanning with bandit
- Code style enforcement with black (line-length: 160)
- Linting with ruff

### Fixed
- **SECURITY**: SQL injection vulnerabilities in db.py (6 locations)
- **SECURITY**: Command injection risks in system.py
- Broken imports in cloud.py, distributed.py, matrix.py (.python → .pythonlib)
- Resource leaks (tracemalloc snapshots, temporary directories)
- Import errors preventing module loading
- Multiple bare except clauses replaced with proper exception handling

### Changed
- All print() calls replaced with proper logging
- type() comparisons replaced with isinstance()
- Module structure improved for better maintainability
- Test coverage improved with additional test cases

### Performance
- pandaslib: optimize_dtypes 2x faster (verified benchmarks)
- pandaslib: nullify_standard_values 200x faster
- pandaslib: get_df_memory_consumption 15x faster
- pandaslib: ensure_float32 5x faster

## [0.1-0.89] - 2024-2026

### Summary
- Internal development versions
- Core functionality development for 31 modules
- Initial test suite creation
- Performance optimizations
- Bug fixes and improvements

---

## Module Categories

### Data Science & Analytics
- pandaslib, polarslib, numpylib, numbalib, matrix

### Database & Storage
- db, redislib, deltalakes, serialization

### Web & Cloud
- web, browser, cloud, graphql

### System & Infrastructure
- system, parallel, monitoring, distributed, scheduling

### Text & NLP
- strings, tokenizers, similarity

### Development Tools
- pythonlib, logginglib, benchmarking, dashlib

### Specialized
- image, filemaker, com, openai
