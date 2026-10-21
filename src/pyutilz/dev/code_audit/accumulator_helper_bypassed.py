"""(internal) accumulator_helper_bypassed now lives in py_ci_shared.code_audit.accumulator_helper_bypassed; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.accumulator_helper_bypassed import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ACCUMULATOR_HINTS,
    _MUTATING_METHODS,
    _SETUP_NAMES,
    _accumulates,
    _bypasses_in,
    _called_names,
    _first_bypass,
    _is_lock_with,
    _is_setup,
    _keys_by_parameter,
    _lines_under_a_lock,
    _module_assigned_names,
    _mutated_structures,
    _owners_of_accumulators,
    scan_accumulator_helper_bypassed,
)

from py_ci_shared.code_audit import accumulator_helper_bypassed as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
