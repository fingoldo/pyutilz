"""(internal) module_docstring now lives in py_ci_shared.code_audit.module_docstring; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.module_docstring import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    scan_duplicate_module_docstring,
)

from py_ci_shared.code_audit import module_docstring as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
