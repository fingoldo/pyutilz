"""(internal) dead_endpoint_params now lives in py_ci_shared.code_audit.dead_endpoint_params; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.dead_endpoint_params import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _REQUEST_SOURCES,
    _ROUTE_METHODS,
    _annotated_source,
    _call_name,
    _is_route,
    _loaded_names,
    _request_bound_params,
    scan_dead_endpoint_parameters,
)

from py_ci_shared.code_audit import dead_endpoint_params as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
