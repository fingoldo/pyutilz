"""(internal) table_drift now lives in py_ci_shared.code_audit.table_drift; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.table_drift import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _collect_assigned_dictwriters,
    _collect_unbound_dictwriters,
    _dictwriter_drifts,
    _dictwriter_headers,
    _fieldnames_literal,
    _format_field_count,
    _literal_str_list,
    _unpaired_wide_fstrings,
    scan_table_header_row_drift,
)

from py_ci_shared.code_audit import table_drift as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
