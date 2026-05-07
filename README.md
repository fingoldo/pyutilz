# pyutilz

[![CI](https://github.com/fingoldo/pyutilz/workflows/CI/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![codecov](https://codecov.io/gh/fingoldo/pyutilz/branch/main/graph/badge.svg)](https://codecov.io/gh/fingoldo/pyutilz)
[![PyPI](https://img.shields.io/pypi/v/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![Python](https://img.shields.io/pypi/pyversions/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python utilities library covering data-frame ops, databases, web/cloud, system monitoring, parallelism, and a unified async LLM-provider interface. The core has zero hard dependencies; every domain ships as an optional extras group so `pip install pyutilz` stays light and you opt into what you need.

## Installation

```bash
pip install pyutilz                   # core only, no hard deps
pip install pyutilz[pandas]           # pandas + numpy + pyarrow
pip install pyutilz[polars]
pip install pyutilz[database]         # SQLAlchemy + psycopg2 + pymysql
pip install pyutilz[web]              # selenium, requests, undetected-chromedriver
pip install pyutilz[cloud]            # boto3 + google-cloud-storage
pip install pyutilz[nlp]              # spacy + nltk + tiktoken + jellyfish
pip install pyutilz[llm]              # anthropic + google-genai + httpx + tenacity + pydantic
pip install pyutilz[system]           # psutil + numba + GPUtil + tqdm + py-cpuinfo
pip install pyutilz[all]              # everything above
pip install pyutilz[dev]              # pytest, ruff, black, mypy, bandit
```

For development:

```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e ".[all,dev]"
pre-commit install
pytest
```

Requires Python 3.8+. Tested on 3.8 through 3.14.

## Modules

| Sub-package          | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `pyutilz.core`       | Core Python helpers: type handling, object loading, lazy-import proxy, version metadata |
| `pyutilz.data`       | `pandaslib`, `polarslib`, `numpylib`, `numbalib`, `matrix` |
| `pyutilz.database`   | PostgreSQL/MySQL helpers, parameterised queries, identifier validation, Redis, Delta Lake |
| `pyutilz.web`        | HTTP/scraping utilities, browser automation, GraphQL, proxy rotation |
| `pyutilz.cloud`      | S3 and Google Cloud Storage helpers                  |
| `pyutilz.system`     | System info, hardware detection, monitoring with timeouts, parallel execution, distributed coordination |
| `pyutilz.text`       | String processing, similarity metrics, NLP tokenisers |
| `pyutilz.dev`        | Logging, benchmarking, dashboards, FileMaker, Jupyter helpers, meta-test utilities |
| `pyutilz.llm`        | Unified async interface across Anthropic, OpenAI, Google Gemini, DeepSeek, xAI Grok, OpenRouter, Claude Code |

## Quick examples

### DataFrame memory optimisation

```python
from pyutilz.data.pandaslib import optimize_dtypes

df = pd.read_csv("large.csv")
df = optimize_dtypes(df)              # 50-80% memory reduction on typical data
```

### Safe parameterised SQL

```python
from pyutilz.database.db import validate_sql_identifier, safe_execute

table = validate_sql_identifier(user_input)            # raises on injection attempts
rows = safe_execute("SELECT * FROM {} WHERE id = %s", (table, user_id))
```

### LLM provider, unified interface

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")
print(p.last_call_summary())   # cost, tokens, upstream provider, finish reason
```

The same `generate()` / `generate_json()` / `generate_stream()` surface works
across every provider; switch by changing the factory key (`"anthropic"`,
`"openai"`, `"deepseek"`, `"gemini"`, `"xai"`, `"openrouter"`, `"claude-code"`).

### OpenRouter health-aware model selection

```python
from pyutilz.llm import list_openrouter_models

# Cheapest healthy Claude variant under $1/1M input, sorted by uptime
rows = list_openrouter_models(
    name_contains="claude",
    max_input_per_1m=1.0,
    sort_by="uptime",
)
top = rows[0]
print(top["id"], top["health"]["best_uptime_30m"], top["health"]["best_latency_p50_ms"], "ms p50")
```

### Performance monitoring

```python
from pyutilz.system.monitoring import timeout_wrapper

@timeout_wrapper(timeout=10)
def slow_call():
    return requests.get("https://api.example.com/data").json()
```

## Testing

1900+ tests, 79.6% line coverage on `src/pyutilz/`. Live LLM-provider
tests are gated behind `--run-live` and skip by default so CI never
spends real money:

```bash
pytest                                # full suite, ~3 min
pytest tests/test_meta/                # static meta-tests only, ~30 s
pytest --run-live -m live              # live LLM smoke tests (real API calls)
pytest --cov=src/pyutilz --cov-report=term-missing
```

Coverage is uploaded to Codecov on every CI run; see the badge above for
the current figure.

### Static meta-tests

`tests/test_meta/` is a 22-test static-check suite catching package-level
drift without exercising runtime behaviour. Wired into
`.pre-commit-config.yaml`, so configuration regressions are caught at
commit time. Highlights:

| Test                                       | Polices                                                                                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_provider_registration.py`            | Every canonical name in `llm.factory._PROVIDER_MODULES` resolves; every alias has a target; no key collisions.                                                       |
| `test_module_alias_integrity.py`           | The 24-entry backward-compat module alias map imports cleanly and proxies real symbols.                                                                              |
| `test_provider_contract.py`                | Every concrete LLM provider inherits from `LLMProvider`, overrides every abstract method, and signature-matches the base interface.                                  |
| `test_optional_deps_isolation.py`          | `import pyutilz` succeeds with each optional-dep group masked; sub-process isolated.                                                                                 |
| `test_no_top_level_side_effects.py`        | Importing pyutilz performs zero network I/O at module-load time. Sub-process socket block.                                                                           |
| `test_api_stability.py`                    | Snapshots the public surface (top-level `__all__`, alias map, public symbol set with signatures, class MROs). Renames / removals fail; additions are silent.         |
| `test_resource_handle_safety.py`           | Every `open()` / `Popen()` / `NamedTemporaryFile()` call is context-managed.                                                                                         |
| `test_encoding_consistency.py`             | Every builtin `open(...)` in production code passes `encoding=` (Windows cp1251 safety).                                                                             |
| `test_no_unicode_in_console_output.py`     | Snapshot-based check for non-ASCII string literals in `print(...)` / `logger.*(...)` calls (Windows stdout safety).                                                  |
| `test_provider_cache_concurrency.py`       | 20 concurrent `get_llm_provider()` callers share one instance; constructor runs exactly once.                                                                        |
| `test_no_import_cycles.py`                 | Tarjan's SCC over the AST-built import graph; flags multi-node cycles.                                                                                               |
| `test_logger_lazy_formatting.py`           | Logger calls use `%`-style formatting (lazy) instead of f-strings (eager) so messages aren't formatted when level is disabled.                                       |
| `test_deferred_drift.py`                   | Counts every `_USER_DEFERRED_*` whitelist across the meta-test suite. Fails when a whitelist grows; refresh via `--refresh-debt-baseline`.                           |

Each meta-test exposes one or both whitelists at file scope:

- `_KNOWN_*` — items consumed via routes static analysis can't see; cite the consumer location.
- `_USER_DEFERRED_*` — items the maintainer surfaced and chose to defer cleanup on. Drain to zero over time.

Shared helpers (`consumer_corpus`, `public_top_level_symbols`,
`capture_signature`, `count_user_deferred_entries`, etc.) live in
[`pyutilz.dev.meta_test_utils`](src/pyutilz/dev/meta_test_utils.py).

### Live LLM tests

Live tests (`tests/test_llm_live.py`) hit real provider APIs and cost a
fraction of a cent per run. Setup:

1. Copy `.env.example` to `.env` and fill in the keys you have. Fixtures
   skip individually when a key is missing, so contributors with a subset
   of accounts still get partial coverage.
2. Run `pytest --run-live tests/test_llm_live.py`. Each test asserts
   `assert_under_budget` ($0.005 cap by default) so an accidental huge
   prompt fails the test rather than burning credits.

`.env` is gitignored; the [detect-secrets](https://github.com/Yelp/detect-secrets)
pre-commit hook blocks accidental commits of API keys to source files.

## Performance

Verified speedups versus the previous in-tree implementations:

| Operation                              | Before  | After  | Speedup |
| -------------------------------------- | ------- | ------ | ------- |
| `optimize_dtypes` (10k × 100)          | ~0.3s   | 0.154s | 2x      |
| `nullify_standard_values`              | ~1.0s   | 0.005s | 200x    |
| `get_df_memory_consumption`            | ~1.1s   | 0.074s | 15x     |
| `ensure_float32` (1k × 60)             | ~0.05s  | 0.010s | 5x      |

OpenRouter `list_openrouter_models()` health-aware lookup: 8x cold-run
speedup (58s → 7s) via TTL cache + shared `httpx.Client` connection pool
+ filter-before-fan-out.

## Security

- Database operations use parameterised queries; `validate_sql_identifier`
  rejects identifiers that don't match `^[A-Za-z_][A-Za-z0-9_]*$`.
- `subprocess` calls never pass `shell=True`.
- Bandit security scans run on every CI build with zero HIGH-severity
  findings.
- API keys for LLM providers are read from `.env` via `pydantic-settings`;
  the file is gitignored and the detect-secrets pre-commit hook blocks
  accidental in-source commits.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
(`black` + `ruff`, line length 160), testing requirements, and the
pull-request process.

## Deferred work

### Meta-test infrastructure

- Standalone GitHub Actions workflow surfacing meta-test status as a
  separate PR badge (currently runs only via pre-commit locally).
- Recurring auto-PR scanning every `_USER_DEFERRED_*` set across the
  meta-test suite, sorted by ease-of-fix. The `test_deferred_drift.py`
  tracker catches *growth*; an active drain loop would shrink the debt.
- Mutation testing on the meta-tests themselves
  (`mutmut run --paths-to-mutate src/pyutilz/llm/ tests/test_meta/`) to
  surface assertions whose value doesn't actually depend on what's being
  checked.

### LLM provider matrix (deferred L-effort items)

The 2026-05-07 audit completed bug fixes (Audit A), info-completeness
correctness (Audit C top-6 — Anthropic real `count_tokens`, cache fields,
ratelimit headers; Claude Code `ResultMessage.usage`; streaming usage;
Gemini `safety_ratings` / grounding / function-calls), Phase 4-A
(OpenRouter `cache_discount` / `is_byok` / web-search / `cache_control`
passthrough / `/parameters` introspection), and Phase 4-B (`tool_calls`
and `citations` capture in the OpenAI-compat base, xAI live-search,
Gemini multi-candidate, Claude Code `/status`).

Remaining matrix items are entire new API families. Each needs an
explicit shape decision (persistence semantics, polling patterns,
separate auth keys) best made with a concrete use case:

**Anthropic**

- Files API (`/v1/files`) — multimodal upload + cross-call reuse.
- Message Batches API (`/v1/messages/batches`) — 50% pricing discount
  for offline workloads, currently unused. Highest financial ROI of the
  deferred items.

**OpenAI**

- Organisation usage API (`/v1/organization/usage`, `/costs`) — opt-in
  `admin_api_key=` constructor knob. Complements the existing per-call
  rate-limit-header capture.
- Responses API beta (`/v1/responses`) — newer surface with server-side
  tool loops.
- Batches API (`/v1/batches`) — same 50% discount story as Anthropic.
- Files API (`/v1/files`).

**Gemini**

- `cachedContents` API full lifecycle (create / list / get / delete).
  Gemini's 90% input-token discount is the largest unrealised cost saving
  in this provider; we currently only thread the resource name through
  `generate()`.
- Files API — required for caching large PDFs / videos before referencing
  them by URI.

**DeepSeek**

- FIM endpoint (`/beta/completions` with prefix + suffix) — unblocks
  IDE-plugin use cases.

**xAI**

- Deferred chat completions — async-poll variant for very long generations.
- Image generation (`grok-2-image` and successors) — different endpoint
  family; would extend pyutilz to multimodal.

**Out of scope (by design)**

- Anthropic Admin API `/cost_report` — needs a separate `sk-ant-admin-` key.
- Gemini Cloud Billing API — separate GCP service-account auth.
- OpenAI deprecated `/v1/dashboard/billing/credit_grants` — endpoint
  removed.
- xAI management API balance — does not exist at present.
- OpenRouter `/credits/coinbase` — niche crypto top-up.

When picking up any of these, start by writing the minimal motivating
use case so the wrapper shape is grounded rather than speculative.

## License

MIT — see [LICENSE](LICENSE).

## Links

- Source: https://github.com/fingoldo/pyutilz
- Issues: https://github.com/fingoldo/pyutilz/issues
- Changelog: [CHANGELOG.md](CHANGELOG.md)
