"""(internal) asymmetric_except_siblings now lives in py_ci_shared.code_audit.asymmetric_except_siblings; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.asymmetric_except_siblings import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CATCH_ALL,
    _RECOVERY_HINTS,
    _asymmetry_finding,
    _class_recovery_sites,
    _guards_compatibly,
    _handler_exception_names,
    _methods_of,
    _recovery_calls,
    scan_asymmetric_except_siblings,
)

from py_ci_shared.code_audit import asymmetric_except_siblings as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
