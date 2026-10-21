"""(internal) uncached_constant_cost_probe now lives in py_ci_shared.code_audit.uncached_constant_cost_probe; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.uncached_constant_cost_probe import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _PROBES,
    _dotted_name_or_partial,
    _has_module_level_memo,
    _matching_probe,
    _probe_call,
    _probe_imports,
    _takes_only_config_parameters,
    scan_uncached_constant_cost_probe,
)

from py_ci_shared.code_audit import uncached_constant_cost_probe as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
