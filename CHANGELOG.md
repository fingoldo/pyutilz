# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `[dataframes]` extras group (`pandas` + `polars` combined); `[all,dev]` is now the recommended one-line install.
- `pyutilz.dev.code_audit` — AST-based scanner + CLI for four recurring bug classes (mutable defaults, late-binding closures, `x or DEFAULT` footguns, silent broad-except swallows). `python -m pyutilz.dev.code_audit <root>`.
- Kernel-tuning cache rewritten as immutable per-`(host, kernel, code_version)` files — no `filelock`, no lost updates under concurrent writers; legacy monolithic caches migrate automatically.
- `pyutilz.dev.benchmarking` sweeps gained `ranking="robust"` (default): interleaved min-over-reps ranking that stays correct on a contended GPU, where the old mean-based ranking could pick the wrong "fastest" config.
- Unified `get_account_credits()` / `check_account_limits()` across every LLM provider (OpenRouter and DeepSeek implement them; others raise an informative `NotImplementedError`).
- OpenRouter provider: model catalogue browsing, live health pre-flight (`check_model_health`, `is_model_healthy`), full usage/cost capture (`fetch_generation_stats`, `last_call_summary`), authoritative per-request billed cost.
- `thinking=` on `generate()`/`generate_stream()` now accepts `"low"/"medium"/"high"/"minimal"` effort strings (previously bool-only), normalized per-provider.
- `tqdmu_lazy_start()` — starts the elapsed-time bar on first iteration, not construction.

### Changed
- **CI/tooling overhaul**: pre-commit's blocking Ruff gate now matches CI exactly (`--select F,E9`, no ignore list) — previously a looser local ignore let unused-import/star-import/empty-fstring/unused-var debt land on `master` that CI would have caught. Black is applied repo-wide via `scripts/black_filtered_apply.py`, which excludes two Black behaviors project convention keeps by hand (arg/collection-list explosion, blank-line insertion); a dedicated `Black (filtered)` CI workflow + badge tracks it.
- DeepSeek/xAI pricing and model registries refreshed (April 2026); DeepSeek default model switched to `deepseek-v4-flash`.

### Fixed
- 14 accidentally-public leaked-import symbols removed from `pyutilz.image` / `logginglib` / `numpylib` / `parallel` (never intended API; real homes documented in each symbol's replacement).
- `pyutilz/web/browser.py` importing a nonexistent `pyutilz.web.pythonlib` (correct home: `pyutilz.core.pythonlib`).
- SQL-injection and command-injection risks in `db.py` / `system.py` (historical, 0.90).

### Meta-test infrastructure
45 tests across 18 files under `tests/test_meta/` guard package-level invariants that runtime tests don't reach: provider registry/alias integrity, public-API snapshot drift, docstring/annotation coverage, import cycles, encoding safety, resource-handle leaks (`with`-block enforcement), lazy-format logging, dead-helper detection, and more. Each has a line-pinned baseline + `--refresh-*-baseline` flag for intentional changes. Wired into `.pre-commit-config.yaml`.

## [1.0.0] - 2026-02-18

### Added
- ~30 hardware-detection functions migrated from `ml_perf_test` into `system.system`: CPU/GPU/board/BIOS detection (py-cpuinfo, WMI, lscpu, dmidecode, nvidia-smi, CUDA capabilities), power plan + battery + large-pages detection, `UtilizationMonitor` background thread.
- New `[system]` extras: `py-cpuinfo`, `GPUtil`, `xmltodict`, `pypiwin32` (Windows).

### Changed
- `get_system_info()` gained platform-specific fields (WMI/lscpu/dmidecode/nvidia-smi) while preserving existing fields for backward compatibility.

### Removed
- Superseded GPU functions (`compute_total_gpus_ram`, `get_gpuinfo_gpu_info`, `get_gpuutil_gpu_info`, `get_pycuda_gpu_info`) — use `get_nvidia_smi_info()` / `get_gpu_cuda_capabilities()`.

### Testing
20 new hardware-detection tests, 95% passing on Windows (1 Linux-only skip).

## [0.90] - 2026-02-18

First public GitHub release: packaging (`pyproject.toml`), CI/CD, badges, 142-test suite, `black`/`ruff`/`bandit` tooling.

### Fixed
- SQL-injection vulnerabilities in `db.py` (6 locations) and command-injection risk in `system.py`.
- Broken imports in `cloud.py` / `distributed.py` / `matrix.py`.
- Resource leaks (`tracemalloc` snapshots, temp directories) and bare `except` clauses.

### Performance
`pandaslib`: `optimize_dtypes` 2x, `nullify_standard_values` 200x, `get_df_memory_consumption` 15x, `ensure_float32` 5x faster (measured).

## [0.1–0.89] - 2024–2026

Internal development versions: core functionality across 31 modules, initial test suite, ongoing performance work.

---

## Module Categories

### Data Science & Analytics
pandaslib, polarslib, numpylib, numbalib, matrix

### Database & Storage
db, redislib, deltalakes, serialization

### Web & Cloud
web, browser, cloud, graphql

### System & Infrastructure
system, parallel, monitoring, distributed, scheduling

### Text & NLP
strings, tokenizers, similarity

### Development Tools
pythonlib, logginglib, benchmarking, dashlib, code_audit

### Specialized
image, filemaker, com, openai
