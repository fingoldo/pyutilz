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
| `pyutilz.web`        | HTTP/scraping utilities, browser automation, GraphQL, statistical proxy health-tracking |
| `pyutilz.cloud`      | S3 and Google Cloud Storage helpers                  |
| `pyutilz.system`     | System/hardware introspection, monitoring with timeouts, parallel execution, distributed coordination |
| `pyutilz.text`       | String processing, Numba-accelerated similarity, AI-text humanisation, NLP tokenisers |
| `pyutilz.dev`        | Logging, benchmarking, dashboards, FileMaker, Jupyter helpers, meta-test utilities |
| `pyutilz.llm`        | Unified async interface across Anthropic, OpenAI, Google Gemini, DeepSeek, xAI Grok, OpenRouter, Claude Code |

## Quick examples

**Shrink a DataFrame's memory** — auto-downcast every column to the
narrowest type that holds the data without precision loss; typical
50-80% reduction on real-world tabular data:

```python
from pyutilz.data.pandaslib import optimize_dtypes
df = optimize_dtypes(df)
```

**Pick the best on-disk format** — measures write/read time and file
size across every parquet/feather/pickle × snappy/lz4/zstd/gzip combo
on the head of the frame, sorted by your chosen metric:

```python
from pyutilz.data.pandaslib import benchmark_dataframe_compression
ranked = benchmark_dataframe_compression(df, head=100_000, sort_by="mean_write_size")
```

**Profile a DataFrame in one call** — per-column dtype, null/unique
counts, value distribution, automatic categorical detection. Works on
pandas and polars frames:

```python
from pyutilz.data.pandaslib import showcase_df_columns
showcase_df_columns(df, max_cat_uniq_qty=50, dropna=False)
```

**Unified LLM interface across 7 providers** — same `generate()` /
`generate_json()` / `generate_stream()` / `get_account_credits()` /
`check_account_limits()` surface; switch by changing one string:

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")

print(p.last_call_summary())
# {'generation_id': 'gen-...', 'upstream_provider': 'Anthropic',
#  'cost_usd': 0.0042, 'input_tokens': 1200, 'cache_hit_tokens': 800,
#  'native_finish_reason': 'end_turn', 'is_byok': False, ...}
```

Streaming preserves token-usage tracking. Account credits work where
the upstream API exposes them (OpenRouter, DeepSeek); other providers
fall back to capturing `anthropic-ratelimit-*` / `x-ratelimit-*`
response headers automatically.

**OpenRouter health-aware model selection** — two-stage lookup
(offline catalogue → concurrent live `/endpoints` health check) drops
degraded upstreams and ranks by live latency. Stage-2 is auth-gated
but not billed:

```python
from pyutilz.llm import list_openrouter_models

# Cheapest healthy Claude variant under $1/1M input, sorted by uptime.
rows = list_openrouter_models(
    name_contains="claude",
    max_input_per_1m=1.0,
    sort_by="uptime",
    min_uptime=0.99,
)
top = rows[0]
print(top["id"], top["health"]["best_uptime_30m"], top["health"]["best_latency_p50_ms"], "ms p50")
```

**Statistical proxy health tracking** — bans a port only when its
error rate is `ban_rate_multiplier` × the cohort average (computed
across peers with enough data). Survives noisy proxies that
occasionally fail and bans ports that genuinely broke:

```python
from pyutilz.web.proxy.base import PortHealthTracker

tracker = PortHealthTracker(min_requests=30, ban_rate_multiplier=2.0,
                             ban_duration=900.0)
tracker.report_success(port_offset=1)
tracker.report_error(port_offset=2)
port = tracker.pick_port(port_range=10_000)        # random non-banned offset
print(tracker.stats())                              # banned_count + averages
```

**Strip the AI fingerprint from generated text** — replaces em-dashes,
hedging openers ("Certainly!"), parenthetical justifications,
overused vocabulary ("delve into", "leverage", "in conclusion"):

```python
from pyutilz.text.humanizer import humanize, strip_ai_patterns, introduce_typos

cleaned = strip_ai_patterns(llm_output)
typo_aug = introduce_typos(cleaned, count=3)        # adversarial dataset aug
print(humanize(llm_output, typo_count=2))            # full pipeline
```

**Numba-accelerated similarity at scale** — pre-pack a tokenised
corpus once, then run repeated batch queries with no Python overhead
in the hot loop:

```python
from pyutilz.text.similarity import SentenceSimilarityIndex

# candidates are already tokenised: list[list[str]]
tokenised = [s.split() for s in corpus]
index = SentenceSimilarityIndex(candidates=tokenised, parallel=True)
scores = index.query("query string here".split())
```

**Parallel apply, RAM-aware worker count** — picks how many processes
fit without OOM-ing the box, then runs the pool with proper exception
propagation:

```python
from pyutilz.system.system import get_max_affordable_workers_count
from pyutilz.system.parallel import applyfunc_parallel

n = get_max_affordable_workers_count(reservedCores=1)
results = applyfunc_parallel(expensive_fn, iterable=inputs, n_cores=n,
                              show_progress=True)
```

**System & hardware introspection in one call** — CPU info (via
py-cpuinfo + WMI on Windows / lscpu on Linux), per-disk free space,
NVIDIA GPU stats, RAM, network interfaces, active power plan; opt-in
flags select what to include:

```python
from pyutilz.system.system import get_system_info

info = get_system_info(
    return_usage_stats=True,
    return_hardware_info=True,
    return_network_info=True,
)
```

**Synchronous timeouts and slow-call alerting:**

```python
from pyutilz.system.monitoring import timeout_wrapper, log_duration

@timeout_wrapper(timeout=10, report_actual_duration=True)
def slow_api_call(): ...

@log_duration(threshold=2.0)           # only logs when call exceeds 2s
def occasionally_slow_function(): ...
```

**Parameterised SQL with identifier validation:**

```python
from pyutilz.database.db import validate_sql_identifier, safe_execute

table = validate_sql_identifier(user_input)             # raises on injection
rows = safe_execute("SELECT * FROM {} WHERE id = %s", (table, user_id))
```

## Security

- Database operations use parameterised queries; `validate_sql_identifier`
  rejects identifiers that don't match `^[A-Za-z_][A-Za-z0-9_]*$`.
- `subprocess` calls never pass `shell=True`.
- Bandit security scans run on every CI build with zero HIGH-severity
  findings.
- LLM API keys are read from `.env` via `pydantic-settings`; the file is
  gitignored and a [detect-secrets](https://github.com/Yelp/detect-secrets)
  pre-commit hook blocks accidental in-source commits.

## Testing

1900+ tests, 79.6% line coverage on `src/pyutilz/`. See
[TESTING.md](TESTING.md) for the static meta-test suite, live LLM
tests, and how to run with coverage.

```bash
pytest                                # full suite, ~3 min
pytest --run-live -m live              # live LLM smoke tests (real API calls, opt-in)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style
(`black` + `ruff`, line length 160), testing requirements, and the
pull-request process.

## Deferred work

### Meta-test infrastructure

- Standalone GitHub Actions workflow surfacing meta-test status as a
  separate PR badge (currently runs only via pre-commit locally).
- Recurring auto-PR scanning every `_USER_DEFERRED_*` set across the
  meta-test suite, sorted by ease-of-fix.
- Mutation testing on the meta-tests themselves to surface assertions
  whose value doesn't actually depend on what's being checked.

### LLM provider matrix

Remaining items from the 2026-05-07 audit are entire new API families;
each needs an explicit shape decision (persistence semantics, polling
patterns, separate auth keys) best made with a concrete use case:

- **Anthropic Files API + Message Batches API** — Batches API gets a
  50% pricing discount on offline workloads; highest financial ROI.
- **OpenAI Organisation usage API** (`/v1/organization/usage`,
  `/costs`) — opt-in `admin_api_key=` knob complementing the existing
  per-call rate-limit-header capture. Plus Responses API beta, Batches
  API, Files API.
- **Gemini `cachedContents` full lifecycle** — Gemini's 90% input-token
  discount is the largest unrealised cost saving in this provider; we
  currently only thread the resource name through `generate()`. Plus
  Files API for caching large PDFs / videos.
- **DeepSeek FIM endpoint** (`/beta/completions` with prefix + suffix)
  for IDE-plugin use cases.
- **xAI deferred chat completions** (async-poll for very long
  generations) and image generation.

Out of scope by design: Anthropic Admin `/cost_report` (needs separate
admin key); Gemini Cloud Billing API (separate GCP service-account
auth); OpenAI deprecated `/credit_grants` (endpoint removed); xAI
management API balance (does not exist); OpenRouter `/credits/coinbase`
(niche).

## License

MIT — see [LICENSE](LICENSE).
