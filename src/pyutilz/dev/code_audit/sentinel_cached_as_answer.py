"""(internal) sentinel_cached_as_answer now lives in py_ci_shared.code_audit.sentinel_cached_as_answer; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sentinel_cached_as_answer import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CACHE_HINTS,
    _WRITE_METHODS,
    _cache_name,
    _cache_write,
    _failure_path_scopes,
    _is_failure_sentinel,
    _sentinel_fallback_arm,
    _statement_value,
    _subscript_cache_write,
    scan_sentinel_cached_as_answer,
)

from py_ci_shared.code_audit import sentinel_cached_as_answer as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
