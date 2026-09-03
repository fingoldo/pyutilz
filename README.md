# pyutilz

[![CI](https://github.com/fingoldo/pyutilz/workflows/CI/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![MyPy](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml/badge.svg)](https://github.com/fingoldo/pyutilz/actions/workflows/mypy-full.yml)
[![Black](https://github.com/fingoldo/pyutilz/workflows/Black/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![codecov](https://codecov.io/gh/fingoldo/pyutilz/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/pyutilz)
[![codecov-numba](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=numba-disabled&label=codecov-numba)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![codecov-full](https://img.shields.io/codecov/c/github/fingoldo/pyutilz/master?flag=combined&label=codecov-full)](https://codecov.io/gh/fingoldo/pyutilz/flags)
[![PyPI](https://img.shields.io/pypi/v/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![Python](https://img.shields.io/pypi/pyversions/pyutilz.svg)](https://pypi.org/project/pyutilz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![docs](https://github.com/fingoldo/pyutilz/actions/workflows/docs.yml/badge.svg)](https://fingoldo.github.io/pyutilz/)

A Python utilities library covering data-frame ops, databases, web/cloud, system monitoring, parallelism, and a unified async LLM-provider interface. The core installs nine hard dependencies -- `numba`, `numpy`, `joblib`, `portalocker`, `psutil`, `pandas`, `tqdm`, `pympler` (plus `tomli` below Python 3.11) -- because `pyutilz.core.pythonlib` and `pyutilz.system.system`, the modules nearly every other subpackage imports, use them unconditionally at import time; `pyproject.toml` records the rationale per dependency. Everything heavier than that (scipy, Pillow, selenium, the cloud SDKs, the LLM SDKs, spaCy, ...) is opt-in through an extras group, so you install only the domains you actually use.

## Installation

```bash
pip install pyutilz[all,dev]          # full install (recommended -- see the note below)

pip install pyutilz                   # core only: numba, numpy, joblib, portalocker, psutil, pandas, tqdm, pympler
pip install pyutilz[dataframes]       # pandas + pyarrow + polars
pip install pyutilz[database]         # SQLAlchemy + psycopg2 + pymysql + redis
pip install pyutilz[web]              # selenium + undetected-chromedriver + requests + grequests + fake-useragent + curl-cffi
pip install pyutilz[cloud]            # boto3 + google-cloud-storage
pip install pyutilz[nlp]              # spacy (>=3.10 only) + nltk + tiktoken + jellyfish + beautifulsoup4 + inflect + emoji-data-python
pip install pyutilz[llm]              # anthropic + google-genai (>=3.9 only) + httpx + tenacity + pydantic + pydantic-settings + tiktoken
pip install pyutilz[system]           # scipy + Pillow + py-cpuinfo + GPUtil + xmltodict + jellyfish
pip install pyutilz[stats]            # documented empty alias -- pyutilz.stats needs only core numpy
pip install pyutilz[speedups]         # orjson -- drop-in accelerator, every use site falls back to stdlib json without it
pip install pyutilz[dash]             # flask + dash + dash-bootstrap-components (pyutilz.dev.dashlib)
pip install pyutilz[prefect]          # prefect, requests (pyutilz.system.scheduling.prefect)
pip install pyutilz[tensorflow]       # tensorflow (system.parallel.set_tf_gpu only)
pip install pyutilz[gpu]              # cupy -- see the caveat below
pip install pyutilz[docs]             # mkdocs-material, to build this documentation site
pip install pyutilz[dev]              # pytest + pytest-cov + pytest-benchmark + pytest-asyncio + pytest-instafail + pytest-progress + pytest-timeout + pytest-randomly + ruff + black + mypy + bandit + sqlglot
```

`[all]` = `pandas,polars,database,web,cloud,nlp,llm,system,stats,speedups`. It deliberately leaves out four
groups, so `pip install pyutilz[all]` canNOT import `pyutilz.dev.dashlib` or
`pyutilz.system.scheduling.prefect` -- add the extra explicitly (`pip install "pyutilz[all,dash]"`):

- `[dash]` -- a whole flask/dash web-server stack for one notebook-dashboard helper.
- `[prefect]` -- a full workflow-orchestration runtime with a large transitive tree.
- `[tensorflow]` -- a multi-hundred-MB ML framework used by exactly one function.
- `[gpu]` -- PyPI's `cupy` is sdist-only, so this extra triggers a full local NVCC build. Install
  the matching binary wheel yourself instead (`pip install cupy-cuda12x`) and skip the extra;
  nothing breaks without it, since every cupy import site is `try/except`-guarded.

`pyproject.toml`'s `all`/`gpu` comments are the source of truth for these exclusions.

For development:

```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e ".[all,dev]"
pip install -r requirements-dev.txt   # py-ci-shared -- a git checkout, so it cannot live in [dev]
pip install pre-commit vulture        # not declared in any extras group; the hooks need both
pre-commit install
pytest
```

Requires Python 3.8+. Tested on 3.8 through 3.14.

## Modules

| Sub-package          | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `pyutilz.core`       | Core Python helpers: type handling, object loading, lazy-import proxy, version metadata, matrix utilities, FileMaker integration, sidecar-verified `safe_pickle`, content-addressable `disk_cache` |
| `pyutilz.data`       | `pandaslib`, `polarslib`, `numpylib`, `numbalib`, `git_checkpoint_cache` (git-tracked backup + auto-restore for a machine-local cache) |
| `pyutilz.database`   | PostgreSQL/MySQL helpers, parameterised queries, identifier validation, Redis, Delta Lake |
| `pyutilz.web`        | HTTP/scraping utilities, browser automation, GraphQL, statistical proxy health-tracking, `url_guard` SSRF-style URL validation, `cached_client`, Decodo proxy provider |
| `pyutilz.cloud`      | S3 and Google Cloud Storage helpers                  |
| `pyutilz.system`     | System/hardware introspection, monitoring with timeouts, parallel execution, distributed coordination, `gpu_dispatch` backend selection, `resilience` retry/circuit-breaker/dead-letter queue, async `single_flight_cache`, `cli_logging`, hot-reloadable TOML `config` |
| `pyutilz.performance`| Per-host `KernelTuningCache` for dispatching CUDA/numba/cupy kernel variants |
| `pyutilz.text`       | String processing, Numba-accelerated similarity, AI-text humanisation, NLP tokenisers, `secrets_scrub` redaction of credentials in logs/tracebacks |
| `pyutilz.dev`        | Logging, benchmarking, dashboards, Jupyter helpers, meta-test utilities, AST-based `code_audit` bug-class scanner + CLI (95 scanners; `get_scanners()` / the CLI's `--check` choices are the authoritative list, no prose enumeration is complete), `ci_log_analyzer`, `freevar_analysis` refactor planner |
| `pyutilz.llm`        | Unified async interface across Anthropic, OpenAI, Google Gemini, DeepSeek, xAI Grok, OpenRouter, Claude Code |
| `pyutilz.stats`      | Numba-jitted normality testing (D'Agostino K², Anderson-Darling) |

## Quick examples

**Shrink a DataFrame's memory** — auto-downcast every column to the
narrowest type that holds the data without precision loss; typical
50-80% reduction on real-world tabular data (measured 2026-09-02):

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
`generate_json()` / `get_account_credits()` / `check_account_limits()`
surface (plus `generate_stream()` on the OpenAI-compatible providers);
switch by changing one string:

```python
from pyutilz.llm import get_llm_provider

p = get_llm_provider("openrouter", model="anthropic/claude-sonnet-4.6")
text = await p.generate("Summarise this", system="You are concise.")

print(p.last_call_summary())
# {'generation_id': 'gen-...', 'upstream_provider': 'Anthropic',
#  'cost_usd': 0.0042, 'input_tokens': 1200, 'cache_hit_tokens': 800,
#  'native_finish_reason': 'end_turn', 'is_byok': False, ...}
```

Streaming preserves token-usage tracking. `get_account_credits()` works
natively for OpenRouter and DeepSeek only; other providers raise
`NotImplementedError`. `check_account_limits()` is native for OpenRouter,
falls back to captured `anthropic-ratelimit-*` / `x-ratelimit-*` response
headers for Anthropic and DeepSeek, and raises `NotImplementedError` by
design for OpenAI/xAI/Gemini regardless of any headers already captured
(Claude Code shells out to the CLI, no HTTP headers to capture at all).

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
filler phrases ("It's worth noting that", "In conclusion,") and overused
vocabulary ("delve into" → "look into", "leverage" → "use"). Note it does
NOT remove hedging openers ("Certainly!") or parenthetical justifications —
those are not in the pattern table:

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
# iterable is a list of per-call arg tuples (passed to func via starmap);
# return_dataframe=False for scalar/list results (True concatenates
# per-call pandas Series/DataFrame results instead).
results = applyfunc_parallel(iterable=inputs, func=expensive_fn, n_jobs=n,
                              return_dataframe=False)
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

**Per-host kernel-tuning cache** — when a project ships multiple CUDA / numba / cupy variants of the same hot numerical kernel, the "best" choice depends on the live GPU. Hardcoded thresholds stop being correct as soon as the package runs on a different cc. This module stores empirically-measured `(variant, block_size, ...)` decisions per `hw_fingerprint` and dispatches at runtime:

```python
from pyutilz.performance.kernel_tuning import KernelTuningCache, hw_fingerprint

cache = KernelTuningCache.load_or_create()
print(hw_fingerprint())                # "cpu_intel-i7-9700k_gpu_gtx-1050-ti_cc6.1"

# Project-side tuner emits per-region winners; pyutilz only stores them.
cache.update("joint_hist_batched", axes=["n_samples", "joint_size"], regions=[
    {"n_samples_max": 200_000, "joint_size_max": 25, "variant": "shared", "block_size": 256},
    {"n_samples_max": None,    "joint_size_max": None, "variant": "shared", "block_size": 512},  # catch-all
])

# Runtime dispatch:
region = cache.lookup("joint_hist_batched", n_samples=1_000_000, joint_size=100)
launch_kernel(variant=region["variant"], block=region["block_size"])
```

Immutable per-`(host, kernel, code_version)` JSON files under `~/.pyutilz/kernel_tuning/` (override via `$PYUTILZ_KERNEL_CACHE_DIR`) — no `filelock`, no read-modify-write, so concurrent writers can never revert each other's fresher entry. Provenance (CUDA driver/runtime, cupy/numba/numpy versions, GPU summary) auto-stamped; stale entries from upgraded libs are detected via `provenance_changed()`. Concrete consumer: [mlframe MRMR](https://github.com/fingoldo/mlframe) feature selection uses this for joint-histogram CUDA RawKernel dispatch (shared-mem vs global-atomic vs numba.cuda), measured 2.6× cumulative speedup at N=1M, p=30.

**Sidecar-verified pickle load** — a crashed job that gets interrupted
mid-`pickle.dump` leaves a silently-truncated file that loads without
error and fails later, far from the cause. `safe_dump`/`safe_load`
write a SHA-256 sidecar next to every pickle and refuse to load if it's
missing or doesn't match, with an atomic, Windows-safe write path. See
the [safe pickle guide](docs/guides/safe_pickle.md):

```python
from pyutilz.core.safe_pickle import safe_dump, safe_load

safe_dump(model, "model.pkl")   # writes model.pkl + model.pkl.sha256
model = safe_load("model.pkl")  # verifies the sidecar before unpickling
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
rows = safe_execute("SELECT * FROM {} WHERE id = %s".format(table), (user_id,))  # table formatted in AFTER validation; only the value goes through the placeholder
```

## Security

- Database operations use parameterised queries; `validate_sql_identifier`
  rejects identifiers that don't match `^[A-Za-z_][A-Za-z0-9_]*$`.
- `subprocess` calls never pass `shell=True`.
- Bandit (`bandit -ll`) and Vulture dead-code scans run as blocking
  gates both locally (pre-commit) and in CI, triaged to zero findings.
- LLM API keys are read from `.env` via `pydantic-settings`; the file is
  gitignored and a [detect-secrets](https://github.com/Yelp/detect-secrets)
  pre-commit hook blocks accidental in-source commits.

## Testing

Exact test counts and coverage percentages are intentionally not pinned here -- they drift with
every commit and go stale silently. The codecov badges at the top of this file carry the current line
coverage (plain, numba-disabled, and the merged `codecov-full` flag); for the current test count
run `pytest --collect-only -q`. See [TESTING.md](TESTING.md) for the static meta-test suite,
live LLM tests, and how to run with coverage.

```bash
pytest                                 # full suite
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
