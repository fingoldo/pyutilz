"""(internal) signature_gated_kwargs now lives in py_ci_shared.code_audit.signature_gated_kwargs; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.signature_gated_kwargs import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _REPORTING_CALLS,
    _gated_parameter,
    _parameter_set_names,
    _passes_the_value_on,
    _says_something,
    _signature_parameter_names,
    scan_signature_gated_kwarg_drops,
)

from py_ci_shared.code_audit import signature_gated_kwargs as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
