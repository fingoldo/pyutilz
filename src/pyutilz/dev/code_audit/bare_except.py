"""(internal) bare_except now lives in py_ci_shared.code_audit.bare_except; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.bare_except import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _is_bare_except,
    scan_bare_except,
)

from py_ci_shared.code_audit import bare_except as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
