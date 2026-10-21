"""(internal) sql_lint now lives in py_ci_shared.code_audit.sql_lint; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sql_lint import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CAST_RE,
    _JSON_EXTRACT_RE,
    _LIMIT_RE,
    _MIN_MAX_CALL_RE,
    _OFFSET_RE,
    _ORDER_BY_RE,
    _SELECT_RE,
    _docstring_constant_ids,
    _matching_close_paren,
    _string_constants,
    scan_sql_aggregate_before_cast,
    scan_sql_limit_without_order_by,
    scan_sql_offset_pagination,
)

from py_ci_shared.code_audit import sql_lint as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
