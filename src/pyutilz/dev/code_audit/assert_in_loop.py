"""(internal) assert_in_loop now lives in py_ci_shared.code_audit.assert_in_loop; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.assert_in_loop import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _body_accumulates,
    _enclosing_function,
    _is_data_sweep,
    scan_assert_in_loop_reports_only_the_first,
)

from py_ci_shared.code_audit import assert_in_loop as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
