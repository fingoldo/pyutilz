"""(internal) machine_dependent_test now lives in py_ci_shared.code_audit.machine_dependent_test; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.machine_dependent_test import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _FunctionDef,
    _TIMER_FUNCS,
    _WALL_CLOCK_SLEEP_SECONDS,
    _deleted_attributes,
    _elapsed_names_in,
    _guarded_lines,
    _is_duration,
    _is_timer_call,
    _is_timer_difference,
    _own_body_nodes,
    _test_functions,
    _unguarded_first_reads,
    _wall_clock_findings_in_fn,
    _wall_clock_sleeps,
    scan_deleted_attribute_read_unconditionally,
    scan_sleep_then_assert,
    scan_wall_clock_assertion,
)

from py_ci_shared.code_audit import machine_dependent_test as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
