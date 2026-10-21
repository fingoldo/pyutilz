"""(internal) locals_get now lives in py_ci_shared.code_audit.locals_get; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.locals_get import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    scan_locals_get_fragile_lookup,
)

from py_ci_shared.code_audit import locals_get as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
