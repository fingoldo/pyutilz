"""(internal) write_count_from_input now lives in py_ci_shared.code_audit.write_count_from_input; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.write_count_from_input import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _COUNTER_WORDS,
    _FuncDef,
    _INPUT_WORDS,
    _SKIPPING_INSERT,
    _counter_assignments,
    _counter_label,
    _derived,
    _docless,
    _function_findings,
    _helper_of,
    _is_len,
    _is_write_counter,
    _len_calls,
    _measured_names,
    _module_functions,
    _reads,
    _receiving_calls,
    _skips_rows,
    _writes_lossily,
    logger,
    scan_write_counted_from_input,
)

from py_ci_shared.code_audit import write_count_from_input as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
