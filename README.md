# PyUtilz

[![CI](https://github.com/fingoldo/pyutilz/workflows/CI/badge.svg)](https://github.com/fingoldo/pyutilz/actions)
[![codecov](https://codecov.io/gh/fingoldo/pyutilz/branch/main/graph/badge.svg)](https://codecov.io/gh/fingoldo/pyutilz)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/8c06a502bda04f2eba80a74945e1566d)](https://app.codacy.com/gh/fingoldo/pyutilz/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

**Comprehensive Python utilities library for data science, databases, web scraping, and system operations.**

---

## 🚀 Features

### Data Science & Analytics
- **pandaslib**: High-performance pandas operations with 20-200x speedups (verified benchmarks)
- **polarslib**: Polars DataFrame utilities for modern data processing
- **numpylib**: NumPy array operations and optimizations
- **numbalib**: Numba JIT compilation helpers for numerical computing
- **matrix**: Matrix operations and linear algebra utilities

### Database & Storage
- **db**: PostgreSQL/MySQL utilities with **SQL injection protection**
- **redislib**: Redis client helpers for caching and pub/sub
- **deltalakes**: Delta Lake integration for data lakehouse architectures
- **serialization**: Efficient object serialization (pickle, JSON, binary)

### Web & Cloud
- **web**: HTTP utilities, web scraping, API clients
- **browser**: Browser automation with Selenium/undetected-chromedriver
- **cloud**: AWS S3 and Google Cloud Storage utilities
- **graphql**: GraphQL client helpers

### System & Infrastructure
- **system**: System monitoring, profiling, process management
- **parallel**: Multiprocessing, threading, and GPU utilities
- **monitoring**: Performance monitoring, timeout wrappers, alerting
- **distributed**: Distributed computing and heartbeat management
- **scheduling**: Prefect workflow scheduling integration

### Text & NLP
- **strings**: Text processing, normalization, tokenization
- **tokenizers**: NLP tokenization (NLTK, spaCy, custom)
- **similarity**: Text similarity metrics (Jaccard, Levenshtein, embeddings)

### Development Tools
- **pythonlib**: Core Python utilities, type handling, object loading
- **logginglib**: Advanced logging configuration
- **benchmarking**: Performance benchmarking utilities
- **dashlib**: Plotly Dash dashboard helpers

### Specialized
- **image**: Image processing utilities
- **filemaker**: FileMaker database integration
- **com**: COM port utilities (Windows)
- **openai**: OpenAI API helpers

---

## 📦 Installation

### From PyPI
```bash
pip install pyutilz
```

### From Source (Development / Editable Mode)

For the latest changes or development, install from a local git clone in editable mode.
This way, `git pull` will automatically pick up code changes without reinstalling:

```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e ".[system]"
```

### With Optional Dependencies

```bash
# For pandas operations
pip install pyutilz[pandas]

# For database operations
pip install pyutilz[database]

# For web scraping
pip install pyutilz[web]

# For cloud storage
pip install pyutilz[cloud]

# For NLP tasks
pip install pyutilz[nlp]

# For system utilities (hardware detection, monitoring)
pip install pyutilz[system]

# Install everything
pip install pyutilz[all]

# Editable mode with extras (combine as needed)
pip install -e ".[system,pandas]"
```

---

## 🔥 Quick Start

### Example 1: Optimize pandas DataFrame Memory

```python
from pyutilz.pandaslib import optimize_dtypes
import pandas as pd

# Load data
df = pd.read_csv('large_file.csv')
print(f"Original memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Optimize dtypes (can reduce memory by 50-80%)
df_optimized = optimize_dtypes(df)
print(f"Optimized memory: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

### Example 2: Safe Database Operations

```python
from pyutilz.db import validate_sql_identifier, safe_execute

# Protect against SQL injection
table_name = validate_sql_identifier(user_input)  # Raises ValueError if malicious

# Use parameterized queries
sql = "SELECT * FROM {} WHERE user_id = %s"
result = safe_execute(sql, (table_name, user_id))
```

### Example 3: System Monitoring

```python
from pyutilz.system import get_system_info

# Get comprehensive system information
info = get_system_info(
    return_usage_stats=True,
    return_hardware_info=True,
    return_network_info=True
)

print(f"CPU Usage: {info['cpu_current_load_percent']}%")
print(f"RAM Free: {info['ram_free_gb']:.1f} GB")
print(f"GPU RAM: {info.get('gpus_ram_free_gb', 'N/A')} GB")
```

### Example 4: Performance Monitoring with Timeouts

```python
from pyutilz.monitoring import timeout_wrapper

@timeout_wrapper(timeout=10)
def slow_api_call():
    response = requests.get('https://api.example.com/data')
    return response.json()

# Automatically returns None if exceeds 10 seconds
result = slow_api_call()
```

---

## 📚 Module Reference

<details>
<summary><b>Click to expand complete module list (31 modules)</b></summary>

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **pandaslib** | Pandas DataFrame operations | `optimize_dtypes`, `remove_constant_columns`, `remove_stale_columns` |
| **polarslib** | Polars DataFrame utilities | `find_nan_cols`, `find_infinite_cols`, `cast_f64_to_f32` |
| **pythonlib** | Core Python utilities | `ensure_list_set_tuple`, `filter_elements_by_type`, `ObjectsLoader` |
| **strings** | Text processing | `tokenize_source`, `fix_spaces`, `jsonize_attributes` |
| **db** | Database operations | `validate_sql_identifier`, `safe_execute`, `build_upsert_query` |
| **system** | System monitoring | `get_system_info`, `show_tracemalloc_snapshot`, `get_max_affordable_workers_count` |
| **web** | Web scraping | `get_external_ip`, HTTP utilities, API clients |
| **browser** | Browser automation | Selenium wrappers, login helpers |
| **monitoring** | Performance tracking | `timeout_wrapper`, `log_duration` |
| **parallel** | Parallel processing | `mem_map_array`, multiprocessing helpers, GPU selection |
| **distributed** | Distributed computing | Heartbeat management, distributed state |
| **cloud** | Cloud storage | S3 upload/download, GCS utilities |
| **logginglib** | Logging setup | Advanced logging configuration |
| **serialization** | Object persistence | Pickle, JSON, binary serialization |
| **similarity** | Text similarity | Jaccard, Levenshtein, embedding similarity |
| **tokenizers** | NLP tokenization | NLTK, spaCy, custom tokenizers |
| **numbalib** | Numba JIT | Random seed helpers, JIT decorators |
| **numpylib** | NumPy operations | Array utilities, top-k selection |
| **matrix** | Linear algebra | Matrix operations |
| **dashlib** | Dash/Plotly | Dashboard component helpers |
| **graphql** | GraphQL | GraphQL client utilities |
| **image** | Image processing | Image utilities |
| **redislib** | Redis | Redis client helpers |
| **deltalakes** | Delta Lake | Delta Lake integration |
| **filemaker** | FileMaker | FileMaker database integration |
| **benchmarking** | Performance | Benchmarking utilities |
| **scheduling/prefect** | Workflow | Prefect task scheduling |
| **com** | COM ports | Windows COM port utilities |
| **openai** | OpenAI API | OpenAI API helpers |
| **notebook_init** | Jupyter | Notebook initialization |

</details>

---

## 🧪 Testing

**Current Status:** ✅ **142 tests passing** (plus 29 meta-tests under `tests/test_meta/`)

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Run specific test module
pytest tests/test_pandaslib.py -v

# Run only the meta-tests (≈ 1 min)
pytest tests/test_meta/ -p no:randomly

# Run performance benchmarks
python -m tests.benchmark_pandaslib
```

### Static meta-tests (`tests/test_meta/`)

A package-level static-check suite that catches whole classes of drift
without exercising runtime behaviour. Wired into `.pre-commit-config.yaml`,
so misconfigurations are caught at commit time rather than in CI.

| Test | Polices |
|------|---------|
| `test_provider_registration.py` | Every canonical name in `llm.factory._PROVIDER_MODULES` points to an importable module / class; every alias resolves to a real canonical entry; no key collisions. |
| `test_module_alias_integrity.py` | Every value in `_MODULE_ALIASES` (the 24-entry backward-compat map) imports cleanly; `pyutilz.<alias>` proxies resolve a real public symbol; alias keys don't collide with sub-package names; targets stay inside the `pyutilz.` namespace. |
| `test_test_source_parity.py` | Every non-trivial production module has a corresponding `tests/test_<name>.py`. Reverse direction also flags test files without a target module. |
| `test_todo_hygiene.py` | Every `TODO` / `FIXME` / `XXX` / `HACK` comment carries an attribution (assignee in parens, ISO date, or `@assignee`). Un-attributed markers fail unless explicitly grandfathered. |
| `test_dead_helpers.py` | Public `def` / `class` symbols inside `pyutilz/llm/` (the only sub-tree with unambiguous internal helpers) are referenced ≥ 2× in the corpus. |
| `test_api_stability.py` | Captures the public surface (top-level `__all__` + alias map + each alias target's public symbol set with signatures + class MROs) into `_api_snapshot.json`. Renames / removals fail; additions silent. Refresh: `pytest tests/test_meta/test_api_stability.py --refresh-api-snapshot`. |
| `test_lazy_import_safety.py` | `_create_lazy_module` proxy infrastructure: unknown dunder → `AttributeError`, proxy returns the same object as direct import, repeated accesses are consistent. |
| `test_optional_deps_isolation.py` | `import pyutilz` succeeds with every optional-dep group masked (pandas / polars / database / web / cloud / nlp / llm); `pyutilz` / `pyutilz.core` / `pyutilz.text` import with **all** optional deps masked. Each scenario runs in a sub-process. |
| `test_no_top_level_side_effects.py` | Importing pyutilz and any probed sub-module performs zero network I/O at module-load time. Blocks `socket.socket` / `urllib.request.urlopen` in a sub-process; any module attempting a request at import time fails the test. |
| `test_deferred_drift.py` | (A2) Counts entries in every `_USER_DEFERRED_*` / `_GRANDFATHERED` whitelist across every meta-test (via AST), compares against `_debt_baseline.json`. Fails when a whitelist GROWS. Refresh via `--refresh-debt-baseline`. Net counter visible per run. |
| `test_provider_contract.py` | (D1) Every concrete LLM provider in `_PROVIDER_MODULES` inherits from `LLMProvider`, overrides every abstract method (`generate`, `generate_json`, `generate_batch`, `estimate_cost`, `count_tokens`), and signature-compatibility with the base interface (no dropped required parameters). |
| `test_encoding_consistency.py` | (D2) Every builtin `open(...)` in production code passes `encoding=` (or uses `"b"` mode). Critical for Windows cp1251/cp1252 — see `feedback_windows_encoding`. Defers 8 known offenders in `_USER_DEFERRED_CALLS`. |
| `test_provider_cache_concurrency.py` | (D3) 20 concurrent `get_llm_provider("anthropic", ...)` callers share the same instance, the constructor runs exactly once, distinct kwargs produce distinct instances. Catches a future refactor breaking double-checked-locking in `_provider_cache`. |
| `test_public_docstrings.py` | (E1) Snapshot-based check for new public symbols without a docstring. Baseline: 176 undocumented. Additions silent, new violations fail. Refresh: `--refresh-docstring-baseline`. |
| `test_public_annotations.py` | (E2) Snapshot-based check for new public functions without complete type annotations. Baseline: 157 unannotated. Refresh: `--refresh-annotation-baseline`. |
| `test_version_consistency.py` | (E3) `pyutilz.__version__`, `version.py`, and `pyproject.toml::[project].version` must all agree. |
| `test_no_import_cycles.py` | (E4) Tarjan's SCC over the AST-built import graph; flags multi-node cycles. pyutilz currently passes (3 single-node `__init__.py` self-references are intentional re-export patterns; not flagged). |
| `test_no_unicode_in_console_output.py` | (E5) Snapshot-based check for non-ASCII string literals in `print(...)` / `logger.*(...)` calls. Baseline: 27 offenders. Critical for Windows stdout. |
| `test_meta_meta.py` | (F1+F2+F3) Every `pytest.fail(...)` has actionable text; meta-tests don't reach into private internals without a whitelist entry; per-test perf-budget overrides match real test names. |

Shared building blocks (`consumer_corpus`, `public_top_level_symbols`, `capture_signature`, `count_user_deferred_entries`, etc.) live in [`pyutilz.dev.meta_test_utils`](src/pyutilz/dev/meta_test_utils.py) and are reused by mlframe's meta-test suite as well.

Each test exposes one or both of these whitelists at file scope:

- `_KNOWN_*` — items consumed via routes a static check can't see; cite the consumer location.
- `_USER_DEFERRED_*` — items the maintainer surfaced and explicitly chose to defer cleanup on. Drain to zero over time.

To install the pre-commit hook:

```bash
pip install pre-commit && pre-commit install
```

The hook runs the meta-test suite on every commit (≈ 30–60 s). For tight inner-loop work, a `manual`-stage variant skips the sub-process tests (PT-8 / PT-9) and runs in ≈ 5 s:

```bash
pre-commit run --hook-stage=manual pyutilz-meta-tests-static-only
```

---

## 📋 TODO — meta-test infrastructure (deferred)

**B1 — GitHub Actions workflow for meta-tests.** Currently the suite runs locally via `.pre-commit-config.yaml`. A standalone CI job with its own status badge would surface drift in PR reviews directly. Useful for downstream contributors who don't have pre-commit installed.

**B2 — Auto-PR for `_USER_DEFERRED_*` drain.** Monthly cron-agent that scans every `_USER_DEFERRED_*` / `_GRANDFATHERED` set across the meta-test suite, sorts by ease-of-fix, and opens a tracking issue. The `tests/test_meta/test_deferred_drift.py` tracker catches *growth* of deferred items, but a recurring punch-list would actively drive cleanup. Currently 12 deferred items across 3 whitelists.

**G — Mutation testing on the meta-tests.** `mutmut run --paths-to-mutate src/pyutilz/llm/ tests/test_meta/` would surface meta-tests whose assertions don't actually depend on the value being checked (false-confidence). Likely surfaces 5–15 weak spots; high quality-bar but low immediate priority.

---

## 📈 Performance Benchmarks

Verified performance improvements (see [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)):

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| `optimize_dtypes` (10k×100) | ~0.3s | 0.154s | **2x faster** |
| `nullify_standard_values` | ~1.0s | 0.005s | **200x faster** |
| `get_df_memory_consumption` | ~1.1s | 0.074s | **15x faster** |
| `ensure_float32` (1k×60) | ~0.05s | 0.010s | **5x faster** |

---

## 🔒 Security

- ✅ **SQL injection protection** - All database operations use parameterized queries or validation
- ✅ **Command injection protection** - No `shell=True` in subprocess calls
- ✅ **Bandit security scans** - Automated security analysis in CI/CD
- ✅ **0 HIGH-severity issues** - Verified by bandit scanner

See [security scan results](.github/workflows/ci.yml) in CI pipeline.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup instructions
- Code style guidelines (Black + Ruff)
- Testing requirements
- Pull request process

Quick start for contributors:
```bash
git clone https://github.com/fingoldo/pyutilz.git
cd pyutilz
pip install -e .[all,dev]
pytest
```

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

**Latest:** v0.90 (2026-02-18)
- Public GitHub release with full packaging
- 142 comprehensive tests
- CI/CD automation with GitHub Actions
- Security hardening (SQL injection fixes, bandit scans)
- Performance optimizations (20-200x improvements)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Anatoly Alexeev

---

## 🔗 Links

- **Documentation:** https://pyutilz.readthedocs.io/ (coming soon)
- **PyPI Package:** https://pypi.org/project/pyutilz/ (coming soon)
- **Issue Tracker:** https://github.com/fingoldo/pyutilz/issues
- **Source Code:** https://github.com/fingoldo/pyutilz
- **Refactoring Summary:** [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

---

## 🙏 Acknowledgments

Built with:
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [Polars](https://www.pola.rs/) - Modern DataFrames
- [NumPy](https://numpy.org/) - Numerical computing
- [Numba](https://numba.pydata.org/) - JIT compilation
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database toolkit
- [Selenium](https://www.selenium.dev/) - Browser automation

---

**⭐ Star this repo if you find it useful!**
