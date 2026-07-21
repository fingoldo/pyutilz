"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS
from .mutable_defaults import scan_mutable_defaults, scan_parameter_aliasing_mutation
from .closures import scan_late_binding_closures
from .default_via_or import scan_default_via_or_trap
from .broad_except import scan_broad_except_swallows
from .nan_equality import scan_nan_equality
from .mutation_during_iteration import scan_mutation_during_iteration
from .sql_lint import scan_sql_limit_without_order_by, scan_sql_offset_pagination
from .dead_cli_flags import scan_dead_cli_flags
from .silent_escalation import scan_log_only_except
from .sql_migrations import scan_sql_migration_idempotency
from .duplicate_conditions import scan_duplicate_conditions
from .missed_await import scan_missed_await, scan_sync_blocking_in_async
from .redundant_test_fit import scan_redundant_test_fit_calls
from .undeclared_imports import scan_undeclared_imports
from .vacuous_assertions import scan_vacuous_assertions
from .locals_globals_output import scan_locals_globals_as_output
from .network_timeout import scan_missing_network_timeout
from .retry_loops import scan_retry_loops
from .module_docstring import scan_duplicate_module_docstring
from .unraised_exceptions import scan_unraised_exceptions
from .credential_logging import scan_credential_shaped_log_args

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
    selected = list(SCANNERS) if checks is None else list(checks)
    out: list[Finding] = []
    for name in selected:
        if name not in SCANNERS:
            raise ValueError(f"unknown check {name!r}; available: {sorted(SCANNERS)}")
        out.extend(SCANNERS[name](root, exclude_dirs=exclude_dirs))
    sev_order = {"P0": 0, "P1": 1, "P2": 2, "Low": 3}
    out.sort(key=lambda f: (sev_order.get(f.severity, 99), f.check, f.file, f.line))
    return out
