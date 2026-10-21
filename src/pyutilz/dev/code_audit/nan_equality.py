"""(internal) nan_equality now lives in py_ci_shared.code_audit.nan_equality; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.nan_equality import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _is_nan_constant,
    scan_nan_equality,
)

from py_ci_shared.code_audit import nan_equality as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
