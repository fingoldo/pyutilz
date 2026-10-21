"""(internal) dead_wiring now lives in py_ci_shared.code_audit.dead_wiring; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.dead_wiring import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _collect_top_level_functions,
    _names_mentioned,
    _propagate_live,
    _seed_live,
    scan_dead_public_callables,
)

from py_ci_shared.code_audit import dead_wiring as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
