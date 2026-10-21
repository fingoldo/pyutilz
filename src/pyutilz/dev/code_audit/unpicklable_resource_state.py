"""(internal) unpicklable_resource_state now lives in py_ci_shared.code_audit.unpicklable_resource_state; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.unpicklable_resource_state import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _AMBIGUOUS_CTORS,
    _OPEN_BUILTIN,
    _RESOURCE_MODULES,
    _UNPICKLABLE_CTORS,
    _UNPICKLABLE_DOTTED_CTORS,
    _class_defines_getstate,
    _find_unpicklable_self_assignments,
    _is_unpicklable_ctor_call,
    _last_component,
    _resource_imported_names,
    _self_attr_target,
    scan_unpicklable_resource_state,
)

from py_ci_shared.code_audit import unpicklable_resource_state as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
