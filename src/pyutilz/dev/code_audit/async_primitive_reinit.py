"""(internal) async_primitive_reinit now lives in py_ci_shared.code_audit.async_primitive_reinit; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.async_primitive_reinit import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_PRIMITIVE_NAMES,
    _assignment_pairs,
    _asyncio_bindings,
    _attribute_assigned_primitive_calls,
    _fanout_limiter_calls,
    _is_asyncio_primitive_call,
    _is_persistent_target,
    scan_async_primitive_reinit_per_call,
)

from py_ci_shared.code_audit import async_primitive_reinit as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
