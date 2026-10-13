"""PyUtilz dev subpackage."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .attempt_archive import archive_provider, fetch_missing_generation_stats
    from .benchmarking import benchmark_algos_by_runtime, sweep_backend_crossover, sweep_backend_grid
    from .blast_radius import assert_only_target_changed, records_snapshot, sql_snapshot
    from .block_extract import apply_extraction, plan_extraction, rename_in_function
    from .dashlib import create_tabs
    from .logginglib import debugged, init_logging, initialize_function_log, log_activity, log_loaded_rows, log_result, log_results, log_throttle_count, log_throttled, logged, reset_log_throttles
    from .meta_test_utils import assert_fields_roundtrip, capture_module_surface, clear_repo_scan_caches, consumer_corpus, count_user_deferred_entries, enumerate_test_files, findings_ratchet, optional_scalar_fields, public_top_level_symbols, safe_import, snake_case_variants_of, strip_lineno, unbacked_audit_dispositions
    from .persistence_sweep import field_sentinels, file_sentinel_locations, jsonl_records, records_holding, rows_holding, sentinel_locations, text_columns, unaccounted_fields, written_files
    from .signature_models import model_from_signature, signature_drift

# Resolved on first attribute access by __getattr__ (lazy: the defining modules may need optional extras).
_REEXPORTS = {
    "archive_provider": "attempt_archive",
    "fetch_missing_generation_stats": "attempt_archive",
    "benchmark_algos_by_runtime": "benchmarking",
    "sweep_backend_crossover": "benchmarking",
    "sweep_backend_grid": "benchmarking",
    "assert_only_target_changed": "blast_radius",
    "records_snapshot": "blast_radius",
    "sql_snapshot": "blast_radius",
    "apply_extraction": "block_extract",
    "plan_extraction": "block_extract",
    "rename_in_function": "block_extract",
    "create_tabs": "dashlib",
    "debugged": "logginglib",
    "init_logging": "logginglib",
    "initialize_function_log": "logginglib",
    "log_activity": "logginglib",
    "log_loaded_rows": "logginglib",
    "log_result": "logginglib",
    "log_results": "logginglib",
    "log_throttle_count": "logginglib",
    "log_throttled": "logginglib",
    "logged": "logginglib",
    "reset_log_throttles": "logginglib",
    "assert_fields_roundtrip": "meta_test_utils",
    "capture_module_surface": "meta_test_utils",
    "clear_repo_scan_caches": "meta_test_utils",
    "consumer_corpus": "meta_test_utils",
    "count_user_deferred_entries": "meta_test_utils",
    "enumerate_test_files": "meta_test_utils",
    "findings_ratchet": "meta_test_utils",
    "optional_scalar_fields": "meta_test_utils",
    "public_top_level_symbols": "meta_test_utils",
    "safe_import": "meta_test_utils",
    "snake_case_variants_of": "meta_test_utils",
    "strip_lineno": "meta_test_utils",
    "unbacked_audit_dispositions": "meta_test_utils",
    "field_sentinels": "persistence_sweep",
    "file_sentinel_locations": "persistence_sweep",
    "jsonl_records": "persistence_sweep",
    "records_holding": "persistence_sweep",
    "rows_holding": "persistence_sweep",
    "sentinel_locations": "persistence_sweep",
    "text_columns": "persistence_sweep",
    "unaccounted_fields": "persistence_sweep",
    "written_files": "persistence_sweep",
    "model_from_signature": "signature_models",
    "signature_drift": "signature_models",
}


__all__ = ["logginglib", "benchmarking", "dashlib", "notebook_init", "code_audit", "freevar_analysis", "block_extract", "signature_models", "persistence_sweep", "archive_provider", "fetch_missing_generation_stats", "benchmark_algos_by_runtime", "sweep_backend_crossover", "sweep_backend_grid", "assert_only_target_changed", "records_snapshot", "sql_snapshot", "apply_extraction", "plan_extraction", "rename_in_function", "create_tabs", "debugged", "init_logging", "initialize_function_log", "log_activity", "log_loaded_rows", "log_result", "log_results", "log_throttle_count", "log_throttled", "logged", "reset_log_throttles", "assert_fields_roundtrip", "capture_module_surface", "clear_repo_scan_caches", "consumer_corpus", "count_user_deferred_entries", "enumerate_test_files", "findings_ratchet", "optional_scalar_fields", "public_top_level_symbols", "safe_import", "snake_case_variants_of", "strip_lineno", "unbacked_audit_dispositions", "field_sentinels", "file_sentinel_locations", "jsonl_records", "records_holding", "rows_holding", "sentinel_locations", "text_columns", "unaccounted_fields", "written_files", "model_from_signature", "signature_drift"]


def __getattr__(name):
    """Bind this package's submodules on attribute access (PEP 562), matching `pyutilz.performance`.

    Without it the three spellings of the same import disagree: `from pyutilz.dev import logginglib` and
    `from pyutilz.dev import *` both work, while `import pyutilz.dev as p; p.logginglib` raises AttributeError -- so a
    downstream module that only ever does the third fails at runtime with no import-time warning.
    Lazy rather than an eager `from . import ...` so importing the package costs nothing until a
    submodule is actually touched (several pull optional third-party dependencies).
    """
    if name in _REEXPORTS:
        import importlib

        value = getattr(importlib.import_module(f".{_REEXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value

    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
