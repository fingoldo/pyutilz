"""(internal) registry now lives in py_ci_shared.code_audit.registry; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.registry import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    OPT_IN_ONLY,
    SHARED_GATE_COUNTERPARTS,
    _BUILTIN_SCANNERS,
    _CHECK_ALIASES,
    _MIN_SCANNERS_FOR_PARALLEL,
    _MIN_SCANNERS_PER_WORKER,
    _SCANNERS,
    _consumer_kwargs,
    _is_picklable,
    _module_registered,
    _physical_cpu_count,
    _run_one,
    _run_scanner,
    _run_scanners_parallel,
    _select_scanners,
    describe_scanners,
    get_check_aliases,
    get_scanners,
    logger,
    register_check_alias,
    register_scanner,
    render_check_catalogue,
    resolve_check,
    run_all,
    run_all_checked,
)

from py_ci_shared.code_audit import registry as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
