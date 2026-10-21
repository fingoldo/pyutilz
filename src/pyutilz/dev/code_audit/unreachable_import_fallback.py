"""(internal) unreachable_import_fallback now lives in py_ci_shared.code_audit.unreachable_import_fallback; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.unreachable_import_fallback import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _catches_import_error,
    _is_type_checking_guard,
    _package_of,
    _resolved_module,
    _statement_modules,
    _statements_in,
    _unconditional_module_imports,
    scan_unreachable_import_fallback,
)

from py_ci_shared.code_audit import unreachable_import_fallback as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
