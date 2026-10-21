"""(internal) readonly_to_numpy_mutation now lives in py_ci_shared.code_audit.readonly_to_numpy_mutation; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.readonly_to_numpy_mutation import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _INPLACE_MUTATORS,
    _is_np_inplace_mutator_call,
    _is_to_numpy_call_without_copy,
    _risky_to_numpy_names,
    scan_readonly_to_numpy_mutation,
)

from py_ci_shared.code_audit import readonly_to_numpy_mutation as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
