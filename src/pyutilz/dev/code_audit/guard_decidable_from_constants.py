"""(internal) guard_decidable_from_constants now lives in py_ci_shared.code_audit.guard_decidable_from_constants; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.guard_decidable_from_constants import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _COMPARISONS,
    _all_bindings,
    _assignment_targets,
    _collect_call_writes,
    _collect_external_write,
    _decided_by,
    _externally_written_names,
    _module_constants,
    scan_guard_decidable_from_constants,
)

from py_ci_shared.code_audit import guard_decidable_from_constants as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
