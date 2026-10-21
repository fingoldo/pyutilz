"""(internal) patch_target_is_a_reexport now lives in py_ci_shared.code_audit.patch_target_is_a_reexport; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.patch_target_is_a_reexport import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _any_calls,
    _bare_calls,
    _index_modules,
    _patch_targets,
    _reexports,
    _resolve,
    _vacuous_reexport_origin,
    scan_patch_target_is_a_reexport,
)

from py_ci_shared.code_audit import patch_target_is_a_reexport as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
