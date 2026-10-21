"""(internal) missed_await now lives in py_ci_shared.code_audit.missed_await; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.missed_await import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BLOCKING_ATTR_CALLS,
    _BLOCKING_ROOT_HINTS,
    _BlockingCallFinder,
    _bare_async_call_findings,
    _is_blocking_call,
    _rebound_names,
    _walk_module_level,
    scan_missed_await,
    scan_sync_blocking_in_async,
)

from py_ci_shared.code_audit import missed_await as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
