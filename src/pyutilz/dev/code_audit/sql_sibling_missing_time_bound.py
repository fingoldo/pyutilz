"""(internal) sql_sibling_missing_time_bound now lives in py_ci_shared.code_audit.sql_sibling_missing_time_bound; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sql_sibling_missing_time_bound import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _KEY_BOUND,
    _LIMITED,
    _SELECT,
    _TIME_BOUND,
    _module_sql,
    scan_sql_sibling_missing_time_bound,
)

from py_ci_shared.code_audit import sql_sibling_missing_time_bound as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
