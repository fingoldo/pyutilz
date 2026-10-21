"""(internal) sql_selects_unread_column now lives in py_ci_shared.code_audit.sql_selects_unread_column; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sql_selects_unread_column import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _FETCH_METHODS,
    _SELECT,
    _SELECT_CALLS,
    _UNPARSEABLE,
    _executed_selects,
    _from_cursor,
    _plain_targets,
    _query_unpack_names,
    _read_names,
    _row_names_of_query,
    _select_columns,
    _unpack_target_and_source,
    _unread_bindings,
    _unread_column_finding,
    scan_sql_selects_unread_column,
)

from py_ci_shared.code_audit import sql_selects_unread_column as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
