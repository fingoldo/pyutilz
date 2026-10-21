"""(internal) asymmetric_resource_guard now lives in py_ci_shared.code_audit.asymmetric_resource_guard; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.asymmetric_resource_guard import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_GUARDABLE_OP_SUFFIXES,
    DEFAULT_GUARD_ATTR_NAMES,
    DEFAULT_GUARD_CALL_NAMES,
    _GuardedCallVisitor,
    _HANDLE_NAME_PARTS,
    _call_shape,
    _is_guard_context_expr,
    _receiver_is_a_handle,
    scan_asymmetric_resource_guard,
)

from py_ci_shared.code_audit import asymmetric_resource_guard as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
