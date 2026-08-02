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
from .bare_except import scan_bare_except
from .console_unicode import scan_console_unicode
from .mojibake import scan_mojibake
from .resource_handle_safety import scan_resource_handle_safety
from .todo_hygiene import scan_todo_hygiene
from .import_cycles import scan_import_cycles

# --- registry -----------------------------------------------------------


SCANNERS: dict[str, Callable[..., list[Finding]]] = {}


def register_scanner(name: str, fn: Callable[..., list[Finding]], *, allow_override: bool = False) -> None:
    """Register a scanner under ``name`` in the shared ``SCANNERS`` registry.

    Raises ``ValueError`` if ``name`` already has a registered scanner, unless
    ``allow_override=True`` -- prevents a downstream project's own scanner (or a stray
    re-import) from silently replacing a built-in check under its name.
    """
    if not allow_override and name in SCANNERS:
        raise ValueError(f"scanner {name!r} is already registered; pass allow_override=True to replace it")
    SCANNERS[name] = fn


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
register_scanner("bare_except", scan_bare_except)
register_scanner("console_unicode", scan_console_unicode)
register_scanner("mojibake", scan_mojibake)
register_scanner("resource_handle_safety", scan_resource_handle_safety)
register_scanner("todo_hygiene", scan_todo_hygiene)
register_scanner("import_cycle", scan_import_cycles)


# Scanners that ``run_all()`` does NOT select by default. Two reasons, both about not breaking a
# downstream project's committed baseline the moment it upgrades pyutilz: several of these need
# project configuration to say anything at all (consumer roots, test roots, filter pairs, the fields
# that belong to somebody else's schema), and the rest report accumulated design debt rather than a
# fresh mistake, which is a decision a project takes deliberately rather than inherits. Name them in
# ``checks=`` to run them.
OPT_IN_ONLY: frozenset[str] = frozenset({
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
    # Similarity is a judgment call, not a fresh mistake: two functions that are 99% alike may be
    # deliberate copy-paste drift worth unifying, or two independently-evolved implementations that
    # happen to still look alike -- a project opts in once it's ready to triage that distinction.
    "near_duplicate_function_body",
})


def get_scanners() -> dict[str, Callable[..., list[Finding]]]:
    """Return a COPY of the scanner registry (mirrors kernel_tuning/registry.py's
    ``get_registry()`` pattern) -- prefer this over importing ``SCANNERS`` directly when you only
    need to read the registry, so an accidental mutation (``get_scanners().pop(...)``) can't
    corrupt the shared dict for every subsequent ``run_all()`` call in the same process."""
    return dict(SCANNERS)


def run_all(
    root: Path,
    checks: Optional[Iterable[str]] = None,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Run every (or selected) scanner against ``root`` and return all
    findings in encounter order. Sort by (severity, check, file, line)
    for stable rendering at the call site."""
    selected = [n for n in SCANNERS if n not in OPT_IN_ONLY] if checks is None else list(checks)
    out: list[Finding] = []
    for name in selected:
        if name not in SCANNERS:
            raise ValueError(f"unknown check {name!r}; available: {sorted(SCANNERS)}")
        out.extend(SCANNERS[name](root, exclude_dirs=exclude_dirs))
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    out.sort(key=lambda f: (sev_order.get(f.severity, 99), f.check, f.file, f.line))
    return out
