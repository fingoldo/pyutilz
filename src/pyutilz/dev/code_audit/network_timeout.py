"""(internal) network_timeout now lives in py_ci_shared.code_audit.network_timeout; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.network_timeout import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _NETWORK_CALL_ATTRS,
    _NETWORK_MODULE_HINTS,
    _call_root_name,
    _directly_imported_network_calls,
    _is_flagged_network_call,
    _resolve_network_callee,
    _timeout_preconfigured_vars,
    scan_missing_network_timeout,
)

from py_ci_shared.code_audit import network_timeout as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
