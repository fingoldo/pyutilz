"""(internal) sentinel_guard_mismatch now lives in py_ci_shared.code_audit.sentinel_guard_mismatch; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sentinel_guard_mismatch import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _FALSY_SENTINELS,
    _assigned_from_call,
    _guard_mismatch_findings,
    _is_none_guards,
    _returns_falsy_from_except,
    _returns_none_anywhere,
    _sentinel_returners_in,
    scan_sentinel_guard_mismatch,
)

from py_ci_shared.code_audit import sentinel_guard_mismatch as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
