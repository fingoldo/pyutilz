"""(internal) partial_fix now lives in py_ci_shared.code_audit.partial_fix; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.partial_fix import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _GUARD_TEMPLATES,
    scan_inconsistent_filter,
    scan_partial_guard_across_siblings,
)

from py_ci_shared.code_audit import partial_fix as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
