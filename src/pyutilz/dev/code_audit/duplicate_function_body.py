"""(internal) duplicate_function_body now lives in py_ci_shared.code_audit.duplicate_function_body; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.duplicate_function_body import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _FuncNode,
    _MIN_BODY_NODES,
    _group_by_body,
    _node_count,
    _normalized_body,
    _stripped_body,
    scan_duplicate_function_body,
)

from py_ci_shared.code_audit import duplicate_function_body as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
