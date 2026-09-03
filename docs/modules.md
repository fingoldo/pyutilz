# Modules

One paragraph per sub-package: what it's for and a motivating example. For the exhaustive API, read the docstrings — this page is about orientation, not exhaustive listing.

## `pyutilz.core`

The lightest subpackage, though not a dependency-free one: type handling, object loading, a lazy-import proxy (so an optional-dependency module can be imported unconditionally and only fails when a member is actually used), version metadata, matrix utilities, FileMaker integration, a content-addressable `disk_cache`, and the [`safe_pickle`](guides/safe_pickle.md) sidecar-verified load/dump pair. Two of its modules import beyond the package's own core dependencies at module level: `matrix.py` needs scipy and `image.py` needs Pillow, both shipped in the `[system]` extra; `pythonlib.py` needs numba, joblib and portalocker, which are core dependencies. Example: `from pyutilz.core.safe_pickle import safe_dump, safe_load` gives every project the same corruption-checked pickle primitive instead of five projects re-implementing sha256 sidecars independently.

## `pyutilz.data`

DataFrame tooling across `pandaslib`, `polarslib`, `numpylib`, and `numbalib`: dtype optimisation (`optimize_dtypes` typically shrinks a frame 50-80%), on-disk format/compression benchmarking (`benchmark_dataframe_compression`), and column profiling (`showcase_df_columns`, works on both pandas and polars). This is the module reached for first whenever a pipeline needs to inspect or shrink a real-world tabular dataset before training. `git_checkpoint_cache` sits alongside them for a different job: it keeps a git-tracked gzip backup of a machine-local cache (`write_json_checkpoint` / `read_json_with_checkpoint_fallback` for an in-memory JSON object, `write_bytes_checkpoint` / `read_bytes_with_checkpoint_fallback` for a cache that is already a file on disk) and auto-restores it when the local copy is wiped or empty.

## `pyutilz.database`

PostgreSQL/MySQL helpers, parameterised queries, SQL identifier validation (`validate_sql_identifier` rejects anything outside `^[A-Za-z_][A-Za-z0-9_]*$`, closing the classic identifier-interpolation injection hole), Redis helpers, and Delta Lake I/O. Example: `safe_execute("SELECT * FROM {} WHERE id = %s".format(table), (user_id,))` where `table` has already passed `validate_sql_identifier` (formatted in after validation; only the value goes through the `%s` placeholder).

## `pyutilz.web`

HTTP/scraping utilities, browser automation (selenium/undetected-chromedriver), GraphQL helpers, and a statistical proxy health-tracker. The tracker (`PortHealthTracker`) bans a proxy port only when its error rate is a configurable multiple of the cohort average computed across peers with enough data — it survives noisy proxies that occasionally fail while still banning ports that genuinely broke, instead of a naive fixed-error-count ban that would be too trigger-happy on a noisy pool. Also here: `url_guard`, the single checked entry point for outbound requests built from untrusted data (SSRF-style host/scheme validation), `cached_client`, a cached and rate-limited HTTP client for polite resumable ingestion, and the Decodo proxy provider under `web/proxy/`.

## `pyutilz.cloud`

S3 and Google Cloud Storage helpers for the common upload/download/list-bucket operations, used wherever a project needs to move artefacts to/from object storage without hand-rolling boto3/`google-cloud-storage` boilerplate per call site.

## `pyutilz.system`

System/hardware introspection (`get_system_info` — CPU via py-cpuinfo/WMI/lscpu, GPU, RAM, disks, network, power plan), timeout-guarded monitoring (`timeout_wrapper`, `log_duration`), RAM-aware parallel execution (`get_max_affordable_workers_count` + `applyfunc_parallel`), and distributed coordination primitives. Smaller members of the same subpackage: `gpu_dispatch` (device selection and shared-memory budgeting queried from the driver rather than a hard-coded table), `resilience` (retry-with-backoff, circuit breaker, dead-letter queue), `single_flight_cache` (per-key async dedupe of concurrent fetches), `cli_logging` (one-line `basicConfig` for standalone entry points) and `config` (thread-safe, hot-reloadable TOML). `pyutilz.performance.kernel_tuning` (a sibling of this concern, split out because kernel tuning is a *performance* topic in its own right — see the [dedicated guide](guides/kernel_tuning_cache.md)) lives adjacent to this package.

## `pyutilz.text`

String processing, Numba-accelerated similarity search (`SentenceSimilarityIndex` pre-packs a tokenised corpus once, then answers repeated batch queries with no per-call Python overhead), AI-text humanisation (`humanize`, `strip_ai_patterns`, `introduce_typos` — strips em-dashes, filler phrases and overused LLM vocabulary — not hedging openers such as "Certainly!", which the pattern table does not cover — useful for adversarial dataset augmentation), NLP tokenisers, and `secrets_scrub` — the canonical redaction helpers for keeping API keys and credentials out of logs and tracebacks.

## `pyutilz.dev`

Logging setup, benchmarking helpers, dashboards, Jupyter notebook helpers, meta-test utilities used by the project's own static test suite, and `code_audit` — an AST-based scanner (+ CLI, `python -m pyutilz.dev.code_audit <root>`) for recurring bug classes (mutable defaults, late-binding closures, broad excepts, non-idempotent SQL migrations, SQL and async hazards, test-quality defects, and more), `ci_log_analyzer` (consolidates FAILED/ERROR lines and pytest warning summaries across every job of one GitHub Actions run) and `freevar_analysis` (AST free-variable analysis for planning a safe function/method extraction out of a monolith). This is the "developer experience" layer — tooling that supports building and testing pyutilz itself and downstream projects, rather than runtime application logic.

The code_audit registry currently holds 95 scanners. The authoritative list is the registry itself — `get_scanners()`, equivalently the `--check` choices in `python -m pyutilz.dev.code_audit --help` — never a prose enumeration: the package docstring's "Implemented checks" bullets and the CLI's one-paragraph `--help` summary are both curated samples, not complete lists, so read them as examples of the shapes covered rather than as the inventory.

## `pyutilz.llm`

A unified async interface across seven LLM providers (Anthropic, OpenAI, Google Gemini, DeepSeek, xAI Grok, OpenRouter, Claude Code) behind one `generate()` / `generate_json()` / `generate_stream()` surface, plus account-credit and rate-limit introspection. See the [dedicated guide](guides/llm_providers.md) for why the abstraction exists and how provider switching works in practice.

## `pyutilz.stats`

Numba-jitted normality testing: D'Agostino K² and Anderson-Darling tests plus a combined `normality_verdict()` helper, for residual-distribution / degenerate-sample audits where the ordinary scipy path is too slow to run per-batch.
