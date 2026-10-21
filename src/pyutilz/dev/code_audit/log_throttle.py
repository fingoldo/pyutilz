"""(internal) log_throttle now lives in py_ci_shared.code_audit.log_throttle; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.log_throttle import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BOUNDED_RETRY_HINTS,
    _LOG_METHODS,
    _STATIC_ITERATION_BOUND,
    _THROTTLE_HINTS,
    _call_name,
    _collect_log_calls,
    _for_loop_looks_bounded_retry,
    _guard_looks_throttled,
    _hot_loop_log_finding,
    _is_log_call,
    _iterable_is_small_literal,
    _looks_like_retry_bound_name,
    _loop_body_has_bounded_retry_break,
    _loop_body_has_meaningful_sleep,
    _loop_looks_bounded_retry,
    _mark_block,
    _single_shot_log_calls,
    _visit_for_loop,
    _visit_if_aware,
    _visit_if_branches,
    _visit_while_loop,
    _while_test_is_unconditionally_true,
    scan_unthrottled_hot_loop_log,
)

from py_ci_shared.code_audit import log_throttle as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
