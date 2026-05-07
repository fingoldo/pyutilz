# Testing

1900+ tests, 79.6% line coverage on `src/pyutilz/`. Live LLM-provider
tests are gated behind `--run-live` and skip by default so CI never
spends real money.

## Running tests

```bash
pytest                                          # full suite, ~3 min
pytest tests/test_meta/                          # static meta-tests only, ~30 s
pytest tests/test_pandaslib.py -v                # one module
pytest --run-live -m live                        # live LLM smoke tests (real API calls)
pytest --cov=src/pyutilz --cov-report=term-missing
```

Coverage is uploaded to Codecov on every CI run.

## Static meta-tests

`tests/test_meta/` is a static-check suite catching package-level
drift without exercising runtime behaviour. Wired into
`.pre-commit-config.yaml`, so configuration regressions are caught at
commit time. Selected entries:

| Test                                    | Polices                                                                                                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_provider_registration.py`         | Every canonical name in `llm.factory._PROVIDER_MODULES` resolves; every alias has a target; no key collisions.                                       |
| `test_module_alias_integrity.py`        | The 24-entry backward-compat module alias map imports cleanly and proxies real symbols.                                                              |
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

## Pre-commit hook

```bash
pip install pre-commit && pre-commit install
```

The hook runs the meta-test suite on every commit (~30-60 s). For
tight inner-loop work, a `manual`-stage variant skips the sub-process
tests and runs in ~5 s:

```bash
pre-commit run --hook-stage=manual pyutilz-meta-tests-static-only
```
