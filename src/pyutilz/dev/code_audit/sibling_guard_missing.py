"""(internal) sibling_guard_missing now lives in py_ci_shared.code_audit.sibling_guard_missing; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sibling_guard_missing import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _MIN_SIBLINGS,
    _VACUOUS_TOKENS,
    _guard_group_findings,
    _leading_guard,
    _names_in,
    _self_attributes,
    _shared_resource,
    _tests_any_of,
    scan_sibling_guard_missing,
)

from py_ci_shared.code_audit import sibling_guard_missing as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
