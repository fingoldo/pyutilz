"""(internal) per_call_state_on_shared_instance now lives in py_ci_shared.code_audit.per_call_state_on_shared_instance; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.per_call_state_on_shared_instance import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _LOCK_HINTS,
    _PERCALL_PREFIXES,
    _collect_shared_class_names,
    _first_arg_name,
    _inherit_shared_bases,
    _mentions_lock,
    _per_call_findings_for_class,
    _registry_container_names,
    _self_attr_loads,
    _self_attr_stores,
    _shared_from_stores_and_caches,
    scan_per_call_state_on_shared_instance,
)

from py_ci_shared.code_audit import per_call_state_on_shared_instance as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
