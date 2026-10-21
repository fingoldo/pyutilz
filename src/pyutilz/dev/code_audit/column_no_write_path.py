"""(internal) column_no_write_path now lives in py_ci_shared.code_audit.column_no_write_path; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.column_no_write_path import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BLOCK_COMMENT,
    _COLUMN,
    _CREATE_TABLE,
    _LINE_COMMENT,
    _NOT_A_COLUMN,
    _PRIMARY_KEY,
    _SELF_FILLING,
    _WRITES,
    _collect_py_corpus,
    _collect_sql_declarations,
    _declared_columns,
    _findings_for_unwritten,
    _select_names,
    _strip_comments,
    _written_names,
    scan_column_no_write_path,
)

from py_ci_shared.code_audit import column_no_write_path as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
