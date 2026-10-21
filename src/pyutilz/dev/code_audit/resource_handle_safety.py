"""(internal) resource_handle_safety now lives in py_ci_shared.code_audit.resource_handle_safety; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.resource_handle_safety import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ancestor_chain_via_parent_map,
    _build_parent_map,
    _is_resource_call,
    _is_under_with_statement,
    scan_resource_handle_safety,
)

from py_ci_shared.code_audit import resource_handle_safety as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
