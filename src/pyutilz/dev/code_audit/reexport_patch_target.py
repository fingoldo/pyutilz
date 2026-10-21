"""(internal) reexport_patch_target now lives in py_ci_shared.code_audit.reexport_patch_target; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.reexport_patch_target import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _PATCH_CALLS,
    _module_aliases,
    _module_file,
    _patched_targets,
    _reexport_origin,
    _reexported_names,
    _references_outside_imports,
    scan_reexport_patch_target,
)

from py_ci_shared.code_audit import reexport_patch_target as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
