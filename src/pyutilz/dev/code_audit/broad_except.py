"""(internal) broad_except now lives in py_ci_shared.code_audit.broad_except; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.broad_except import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BEST_EFFORT_OPS,
    _DOCUMENTED_RATIONALE_MARKERS,
    _LOG_METHODS,
    _handler_has_documented_rationale,
    _has_debug_only_log_call,
    _has_log_call,
    _has_raise,
    _is_broad_except,
    _is_debug_log_stmt,
    _is_default_literal,
    _is_flaggable_handler,
    _is_silent_swallow,
    _is_tolerated_stmt,
    _returns_or_assigns_default,
    _try_body_is_best_effort_op,
    _try_body_is_imports_only,
    scan_broad_except_swallows,
)

from py_ci_shared.code_audit import broad_except as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
