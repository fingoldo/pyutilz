"""(internal) retry_loops now lives in py_ci_shared.code_audit.retry_loops; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.retry_loops import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _except_has_sleep,
    _loop_has_bounding_raise,
    _loop_has_break,
    _own_nodes,
    _try_body_has_blocking_wait_with_timeout,
    _try_has_sleep_or_continue,
    scan_retry_loops,
)

from py_ci_shared.code_audit import retry_loops as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
