"""(internal) shielded_resource_release now lives in py_ci_shared.code_audit.shielded_resource_release; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.shielded_resource_release import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_RELEASE_CALL_NAMES,
    _names_read_in,
    _release_call_args_in_finally_blocks,
    _shielded_closure_name,
    scan_shielded_resource_release_race,
)

from py_ci_shared.code_audit import shielded_resource_release as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
