"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS
from .mutable_defaults import scan_mutable_defaults, scan_parameter_aliasing_mutation
from .closures import scan_late_binding_closures
from .default_via_or import scan_default_via_or_trap
from .broad_except import scan_broad_except_swallows
from .nan_equality import scan_nan_equality
from .mutation_during_iteration import scan_mutation_during_iteration
from .sql_lint import scan_sql_limit_without_order_by, scan_sql_offset_pagination, scan_sql_aggregate_before_cast
from .getattr_unknown_attribute import scan_getattr_unknown_attribute
from .getattr_literal_on_known_dataclass import scan_getattr_literal_on_known_dataclass
from .locals_get import scan_locals_get_fragile_lookup
from .dead_cli_flags import scan_dead_cli_flags
from .silent_escalation import scan_log_only_except
from .sql_migrations import scan_sql_migration_idempotency
from .duplicate_conditions import scan_duplicate_conditions
from .duplicate_function_body import scan_duplicate_function_body
from .near_duplicate_function_body import scan_near_duplicate_function_body
from .missed_await import scan_missed_await, scan_sync_blocking_in_async
from .redundant_test_fit import scan_redundant_test_fit_calls
from .undeclared_imports import scan_undeclared_imports
from .vacuous_assertions import scan_vacuous_assertions, scan_tautological_is_not_none_only_tests
from .locals_globals_output import scan_locals_globals_as_output
from .network_timeout import scan_missing_network_timeout
from .retry_loops import scan_retry_loops
from .module_docstring import scan_duplicate_module_docstring
from .unraised_exceptions import scan_unraised_exceptions
from .credential_logging import scan_credential_shaped_log_args
from .docstring_args import scan_docstring_args_completeness
from .return_annotation import scan_return_annotation_mismatch
from .shielded_resource_release import scan_shielded_resource_release_race
from .duplicate_credential_regex import scan_duplicate_credential_regex
from .asymmetric_resource_guard import scan_asymmetric_resource_guard
from .spy_arity import scan_stale_test_spy_arity
from .log_throttle import scan_unthrottled_hot_loop_log
from .dead_import import scan_possibly_dead_import
from .unpicklable_resource_state import scan_unpicklable_resource_state
from .skip_masking_except import scan_except_skip_masks_call_under_test
from .uncurated_star_export import scan_uncurated_star_exports
from .dead_wiring import scan_dead_public_callables
from .vacuous_matching import scan_vacuous_empty_pattern_match
from .tautological_guard import scan_tautological_guards
from .table_drift import scan_table_header_row_drift
from .provenance_flow import scan_record_field_flow
from .claimed_invariants import scan_unenforced_docstring_invariants
from .partial_fix import scan_partial_guard_across_siblings, scan_inconsistent_filter
from .measurement_hygiene import scan_regex_integer_parse, scan_thresholds_below_documented_result
from .domain_boundary import scan_domain_vocabulary_leak
from .readonly_to_numpy_mutation import scan_readonly_to_numpy_mutation
from .hardcoded_test_path import scan_hardcoded_absolute_path_in_test
from .async_primitive_reinit import scan_async_primitive_reinit_per_call
from .llm_max_tokens_cap import scan_llm_call_missing_max_tokens_cap
from .bare_except import scan_bare_except
from .console_unicode import scan_console_unicode
from .mojibake import scan_mojibake
from .resource_handle_safety import scan_resource_handle_safety
from .todo_hygiene import scan_todo_hygiene
from .import_cycles import scan_import_cycles
from .effect_flag_outside_its_effect import scan_effect_flag_outside_its_effect
from .guard_decidable_from_constants import scan_guard_decidable_from_constants
from .count_then_fetch_same_table import scan_count_then_fetch_same_table
from .accumulator_helper_bypassed import scan_accumulator_helper_bypassed
from .sentinel_cached_as_answer import scan_sentinel_cached_as_answer
from .sql_selects_unread_column import scan_sql_selects_unread_column
from .asymmetric_except_siblings import scan_asymmetric_except_siblings
from .unreachable_import_fallback import scan_unreachable_import_fallback
from .comment_names_missing_symbol import scan_comment_names_missing_symbol, scan_comment_cites_absolute_line
from .unit_suffix_mismatch import scan_unit_suffix_mismatch
from .sentinel_guard_mismatch import scan_sentinel_guard_mismatch
from .stats_key_coverage import scan_stats_key_coverage
from .constructor_param_overwritten import scan_constructor_param_overwritten
from .lazy_log_assertion import scan_lazy_log_assertion
from .raising_stub_swallowed import scan_raising_stub_swallowed
from .per_call_state_on_shared_instance import scan_per_call_state_on_shared_instance
from .uncached_constant_cost_probe import scan_uncached_constant_cost_probe
from .source_text_assertions import scan_source_text_assertions
from .docstring_numbers_moved_to_config import scan_docstring_numbers_moved_to_config

# --- registry -----------------------------------------------------------


_SCANNERS: dict[str, Callable[..., list[Finding]]] = {}


def register_scanner(name: str, fn: Callable[..., list[Finding]], *, allow_override: bool = False) -> None:
    """Register a scanner under ``name`` in the shared scanner registry.

    Raises ``ValueError`` if ``name`` already has a registered scanner, unless
    ``allow_override=True`` -- prevents a downstream project's own scanner (or a stray
    re-import) from silently replacing a built-in check under its name.
    """
    if not allow_override and name in _SCANNERS:
        raise ValueError(f"scanner {name!r} is already registered; pass allow_override=True to replace it")
    _SCANNERS[name] = fn


register_scanner("mutable_default", scan_mutable_defaults)
register_scanner("late_binding_closure", scan_late_binding_closures)
register_scanner("default_via_or", scan_default_via_or_trap)
register_scanner("broad_except_swallow", scan_broad_except_swallows)
register_scanner("nan_equality", scan_nan_equality)
register_scanner("mutation_during_iteration", scan_mutation_during_iteration)
register_scanner("sql_limit_without_order_by", scan_sql_limit_without_order_by)
register_scanner("sql_offset_pagination", scan_sql_offset_pagination)
register_scanner("dead_cli_flag", scan_dead_cli_flags)
register_scanner("log_only_except", scan_log_only_except)
register_scanner("sql_migration_not_idempotent", scan_sql_migration_idempotency)
register_scanner("duplicate_condition", scan_duplicate_conditions)
register_scanner("duplicate_function_body", scan_duplicate_function_body)
register_scanner("near_duplicate_function_body", scan_near_duplicate_function_body)
register_scanner("missed_await", scan_missed_await)
register_scanner("redundant_test_fit_call", scan_redundant_test_fit_calls)
register_scanner("undeclared_import", scan_undeclared_imports)
register_scanner("vacuous_assertion", scan_vacuous_assertions)
register_scanner("locals_globals_as_output", scan_locals_globals_as_output)
register_scanner("missing_network_timeout", scan_missing_network_timeout)
register_scanner("parameter_aliasing_mutation", scan_parameter_aliasing_mutation)
register_scanner("sync_blocking_in_async", scan_sync_blocking_in_async)
register_scanner("retry_loop", scan_retry_loops)
register_scanner("duplicate_module_docstring", scan_duplicate_module_docstring)
register_scanner("unraised_exception_class", scan_unraised_exceptions)
register_scanner("credential_shaped_log_arg", scan_credential_shaped_log_args)
register_scanner("docstring_args_incomplete", scan_docstring_args_completeness)
register_scanner("return_annotation_mismatch", scan_return_annotation_mismatch)
register_scanner("sql_aggregate_before_cast", scan_sql_aggregate_before_cast)
register_scanner("getattr_unknown_attribute", scan_getattr_unknown_attribute)
register_scanner("getattr_literal_on_known_dataclass", scan_getattr_literal_on_known_dataclass)
register_scanner("locals_get_fragile_lookup", scan_locals_get_fragile_lookup)
register_scanner("shielded_resource_release_race", scan_shielded_resource_release_race)
# canonical_module_rel_paths designates THIS package's own scanner-definition modules, PLUS the
# canonical secret-redaction module, as the credential-shaped-regex source of truth:
# credential_logging.py's _CREDENTIAL_NAME_RE and this scanner's own
# DEFAULT_CREDENTIAL_KEYWORDS_RE necessarily contain credential-shaped keywords (that's their
# entire job as SECURITY-SCANNING META-TOOLING, not production redaction/secret-handling logic),
# and text/secrets_scrub.py's regexes are the actual canonical production scrubber every
# downstream project should import instead of writing its own -- without this, the scanner flags
# all three, the only credential-shaped re.compile(...) calls anywhere in this codebase, every run.
register_scanner(
    "duplicate_credential_regex",
    partial(
        scan_duplicate_credential_regex,
        canonical_module_rel_paths=frozenset(
            {
                "dev/code_audit/credential_logging.py",
                "dev/code_audit/duplicate_credential_regex.py",
                "text/secrets_scrub.py",
            }
        ),
    ),
)
register_scanner("asymmetric_resource_guard", scan_asymmetric_resource_guard)
register_scanner("stale_test_spy_arity", scan_stale_test_spy_arity)
register_scanner("unthrottled_hot_loop_log", scan_unthrottled_hot_loop_log)
register_scanner("possibly_dead_import", scan_possibly_dead_import)
register_scanner("unpicklable_resource_state", scan_unpicklable_resource_state)
register_scanner("tautological_is_not_none_only_test", scan_tautological_is_not_none_only_tests)
register_scanner("except_skip_masks_call_under_test", scan_except_skip_masks_call_under_test)
register_scanner("uncurated_star_export", scan_uncurated_star_exports)
register_scanner("dead_public_callable", scan_dead_public_callables)
register_scanner("vacuous_empty_pattern_match", scan_vacuous_empty_pattern_match)
register_scanner("tautological_guard", scan_tautological_guards)
register_scanner("table_header_row_drift", scan_table_header_row_drift)
register_scanner("record_field_flow", scan_record_field_flow)
register_scanner("unenforced_docstring_invariant", scan_unenforced_docstring_invariants)
register_scanner("partial_guard_across_siblings", scan_partial_guard_across_siblings)
register_scanner("inconsistent_filter", scan_inconsistent_filter)
register_scanner("regex_integer_parse_truncation", scan_regex_integer_parse)
register_scanner("threshold_below_documented_result", scan_thresholds_below_documented_result)
register_scanner("domain_vocabulary_leak", scan_domain_vocabulary_leak)
register_scanner("readonly_to_numpy_mutation", scan_readonly_to_numpy_mutation)
register_scanner("source_text_assertion", scan_source_text_assertions)
register_scanner("docstring_numbers_moved_to_config", scan_docstring_numbers_moved_to_config)
register_scanner("raising_stub_swallowed", scan_raising_stub_swallowed)
register_scanner("lazy_log_assertion", scan_lazy_log_assertion)
register_scanner("constructor_param_overwritten", scan_constructor_param_overwritten)
register_scanner("stats_key_coverage", scan_stats_key_coverage)
register_scanner("sentinel_guard_mismatch", scan_sentinel_guard_mismatch)
register_scanner("unit_suffix_mismatch", scan_unit_suffix_mismatch)
register_scanner("comment_names_missing_symbol", scan_comment_names_missing_symbol)
register_scanner("comment_cites_absolute_line", scan_comment_cites_absolute_line)
register_scanner("unreachable_import_fallback", scan_unreachable_import_fallback)
register_scanner("asymmetric_except_siblings", scan_asymmetric_except_siblings)
register_scanner("effect_flag_outside_its_effect", scan_effect_flag_outside_its_effect)
register_scanner("bare_except", scan_bare_except)
register_scanner("console_unicode", scan_console_unicode)
register_scanner("mojibake", scan_mojibake)
register_scanner("resource_handle_safety", scan_resource_handle_safety)
register_scanner("todo_hygiene", scan_todo_hygiene)
register_scanner("import_cycle", scan_import_cycles)
register_scanner("hardcoded_absolute_path_in_test", scan_hardcoded_absolute_path_in_test)
register_scanner("async_primitive_reinit_per_call", scan_async_primitive_reinit_per_call)
register_scanner("llm_call_missing_max_tokens_cap", scan_llm_call_missing_max_tokens_cap)
register_scanner("per_call_state_on_shared_instance", scan_per_call_state_on_shared_instance)
register_scanner("uncached_constant_cost_probe", scan_uncached_constant_cost_probe)
register_scanner("guard_decidable_from_constants", scan_guard_decidable_from_constants)
register_scanner("sql_selects_unread_column", scan_sql_selects_unread_column)
register_scanner("count_then_fetch_same_table", scan_count_then_fetch_same_table)
register_scanner("sentinel_cached_as_answer", scan_sentinel_cached_as_answer)
register_scanner("accumulator_helper_bypassed", scan_accumulator_helper_bypassed)


# Scanners that ``run_all()`` does NOT select by default. Two reasons, both about not breaking a
# downstream project's committed baseline the moment it upgrades pyutilz: several of these need
# project configuration to say anything at all (consumer roots, test roots, filter pairs, the fields
# that belong to somebody else's schema), and the rest report accumulated design debt rather than a
# fresh mistake, which is a decision a project takes deliberately rather than inherits. Name them in
# ``checks=`` to run them.
OPT_IN_ONLY: frozenset[str] = frozenset({
    # Opt-in: 225 hits in one package, and most are legitimate -- coverage annotations in
    # tests ("lines 54-90"), and prose citing a line in a file it is discussing. The rule
    # cannot tell those from a rotted pointer, and the rotted ones are better caught by
    # citing symbols instead, which comment_names_missing_symbol then validates.
    "comment_cites_absolute_line",
    # Opt-in because its precision is not good enough to run unattended, and that was measured rather than
    # guessed. Against four repos it produced three hits, ALL false: two docstrings naming a threshold that
    # belongs to a DIFFERENT function, and one reading "12-permutation" as a tunable because "per-call"
    # earlier in the line matched the keyword list. Its one true positive is a real defect that has since
    # been fixed, so it can only be demonstrated on a reconstruction. Useful pointed at a file you already
    # suspect; not useful in a ratchet, where three false alarms would teach everyone to refresh the
    # baseline without reading it.
    "docstring_numbers_moved_to_config",
    "dead_public_callable",
    "vacuous_empty_pattern_match",
    "tautological_guard",
    "table_header_row_drift",
    "record_field_flow",
    "unenforced_docstring_invariant",
    "partial_guard_across_siblings",
    "inconsistent_filter",
    "regex_integer_parse_truncation",
    "threshold_below_documented_result",
    "domain_vocabulary_leak",
    # Warn-only by design, each for its own reason rather than as a blanket posture:
    # per_call_state_on_shared_instance detects a lock lexically, so a lock taken by the CALLER
    # reads as absent; uncached_constant_cost_probe cannot tell a probe that must be re-taken
    # (a liveness check) from one that must not. Both feed a triage list, not a commit gate.
    "per_call_state_on_shared_instance",
    "uncached_constant_cost_probe",
})


def get_scanners() -> dict[str, Callable[..., list[Finding]]]:
    """Return a COPY of the scanner registry (mirrors kernel_tuning/registry.py's
    ``get_registry()`` pattern) -- the ONLY supported way to read the registry, so an accidental
    mutation (``get_scanners().pop(...)``) can't corrupt the shared dict for every subsequent
    ``run_all()`` call in the same process. Writes go through ``register_scanner()``, whose
    duplicate-name guard a direct ``_SCANNERS[name] = fn`` assignment would bypass."""
    return dict(_SCANNERS)


def _run_one(args: tuple[str, Path, frozenset[str]]) -> list[Finding]:
    """Module-level (picklable) trampoline for ``ProcessPoolExecutor`` -- runs one named
    scanner and returns its findings. A bound/local closure can't be pickled for the
    cross-process call, so this indirection is required, not just style."""
    name, root, exclude_dirs = args
    return _SCANNERS[name](root, exclude_dirs=exclude_dirs)


# Below this many scanners, process-pool startup (import pyutilz + its scanner modules in
# each fresh interpreter) costs more than it saves -- e.g. a single-scanner unit test
# calling run_all(checks=["default_via_or"]) stays sequential and instant.
_MIN_SCANNERS_FOR_PARALLEL = 4

# Each worker pays a FIXED cost that does not shrink as workers are added: a fresh
# interpreter spawn, `import pyutilz.dev.code_audit`, and a full re-parse of the corpus into
# that process's own _PARSE_CACHE (workers share no memory). Measured on glossum_backend_
# scripts' 300-file package: ~1.07s parse + ~0.3s import per worker, against ~45.8s of total
# scan work. So past a point each extra worker costs more than the slice of scan work it
# takes off the critical path, and throughput goes DOWN -- confirmed by a worker-count sweep
# (best-of-2 each, identical 262 findings at every setting):
#     w=4 22.9s | w=6 15.8s | w=8 12.0s | w=10 11.7s | w=12 12.2s | w=16 14.4s | w=22 13.4s
# Giving each worker a batch of at least this many scanners keeps it on the flat part of that
# curve instead of spawning one interpreter per scanner.
_MIN_SCANNERS_PER_WORKER = 5


def _physical_cpu_count() -> int:
    """Physical cores, falling back to ``os.cpu_count()``.

    ``os.cpu_count()`` reports LOGICAL CPUs (22 on the machine measured above, vs 16 physical).
    These scanners are CPU-bound pure-Python AST walks that gain nothing from hyperthread
    siblings, so sizing the pool off the logical count just over-subscribes the physical cores
    while adding another full corpus re-parse per extra worker.
    """
    import os

    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception:  # psutil missing or unable to introspect -- fall back, never fail the scan
        pass
    return os.cpu_count() or 1


def run_all(
    root: Path,
    checks: Optional[Iterable[str]] = None,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    parallel: bool = True,
) -> list[Finding]:
    """Run every (or selected) scanner against ``root`` and return all
    findings in encounter order. Sort by (severity, check, file, line)
    for stable rendering at the call site.

    ``parallel`` (default True): each scanner does its own full-corpus file walk + AST-walk
    over the (parse-cached) tree, and with ~60 registered scanners this is almost entirely
    CPU-bound Python-level work that holds the GIL -- a large repo's full run_all() (e.g.
    glossum_backend_scripts's ~1500 files) took ~95s single-process. Scanners are mutually
    independent (each just appends Findings to its own list), so distributing them across a
    ProcessPoolExecutor is a pure wall-clock win with IDENTICAL output (same findings, same
    final sort) -- confirmed via a byte-for-byte comparison against the sequential path
    before landing this. Each worker process re-parses files into its own _PARSE_CACHE
    (workers don't share memory), a real but much smaller cost than the AST-walk time saved
    by running scanners concurrently across cores.
    """
    selected = [n for n in _SCANNERS if n not in OPT_IN_ONLY] if checks is None else list(checks)
    for name in selected:
        if name not in _SCANNERS:
            raise ValueError(f"unknown check {name!r}; available: {sorted(_SCANNERS)}")

    out: list[Finding] = []
    if parallel and len(selected) >= _MIN_SCANNERS_FOR_PARALLEL:
        import concurrent.futures

        # See _MIN_SCANNERS_PER_WORKER: cap by physical cores AND by "enough scanners per
        # worker to amortize its fixed spawn+import+re-parse cost", never by the raw
        # scanner count (which used to spawn one interpreter per scanner on a big machine).
        max_workers = max(2, min(_physical_cpu_count(), len(selected) // _MIN_SCANNERS_PER_WORKER))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            for findings in pool.map(_run_one, [(name, root, exclude_dirs) for name in selected]):
                out.extend(findings)
    else:
        for name in selected:
            out.extend(_SCANNERS[name](root, exclude_dirs=exclude_dirs))

    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    out.sort(key=lambda f: (sev_order.get(f.severity, 99), f.check, f.file, f.line))
    return out


def __getattr__(name: str) -> dict[str, Callable[..., list[Finding]]]:
    """Deprecated read-only shim for the former public ``SCANNERS`` name (PEP 562).

    Returns a COPY: the registry is now private precisely because a direct handle let a caller
    ``pop()`` a built-in check for the rest of the process, or assign a replacement that
    ``register_scanner``'s duplicate-name guard would have rejected. Use ``get_scanners()`` /
    ``register_scanner()``.
    """
    if name == "SCANNERS":
        import warnings

        warnings.warn(
            "code_audit.registry.SCANNERS is deprecated and returns a copy; " "use get_scanners() to read and register_scanner() to write.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dict(_SCANNERS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
