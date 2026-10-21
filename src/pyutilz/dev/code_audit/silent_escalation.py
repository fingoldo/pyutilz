"""(internal) silent_escalation now lives in py_ci_shared.code_audit.silent_escalation; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.silent_escalation import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_ESCALATION_ATTRS,
    _ESCALATION_METHOD_NAMES,
    _call_escalates,
    _escalates_to,
    _file_uses_escalation_convention,
    _is_log_only_handler,
    _looks_like_error_target,
    _node_escalates,
    _returns_failure_marker,
    _target_name,
    scan_log_only_except,
)

from py_ci_shared.code_audit import silent_escalation as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
