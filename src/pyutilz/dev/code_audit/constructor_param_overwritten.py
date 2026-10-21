"""(internal) constructor_param_overwritten now lives in py_ci_shared.code_audit.constructor_param_overwritten; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.constructor_param_overwritten import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CONFIG_READS,
    _assigns_by_method,
    _config_via_sibling_calls,
    _direct_config_overwrites,
    _overwrite_candidates,
    _param_backed_attributes,
    _reads_config,
    _self_attr_targets,
    scan_constructor_param_overwritten,
)

from py_ci_shared.code_audit import constructor_param_overwritten as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
