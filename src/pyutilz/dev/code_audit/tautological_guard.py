"""(internal) tautological_guard now lives in py_ci_shared.code_audit.tautological_guard; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.tautological_guard import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ORDERING,
    _classify_and_operands,
    _root_of,
    _target_key,
    scan_tautological_guards,
)

from py_ci_shared.code_audit import tautological_guard as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
