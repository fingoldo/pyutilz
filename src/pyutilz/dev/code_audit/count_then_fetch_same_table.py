"""(internal) count_then_fetch_same_table now lives in py_ci_shared.code_audit.count_then_fetch_same_table; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.count_then_fetch_same_table import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _COUNT,
    _EXECUTORS,
    _GROUPED,
    _PAGINATED,
    _count_fetch_finding,
    _first_redundant_fetch_line,
    _outer_query,
    _queries_in,
    scan_count_then_fetch_same_table,
)

from py_ci_shared.code_audit import count_then_fetch_same_table as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
