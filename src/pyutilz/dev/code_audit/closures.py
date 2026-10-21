"""(internal) closures now lives in py_ci_shared.code_audit.closures; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.closures import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _STORING_METHODS,
    _SYNCHRONOUS_CONSUMERS,
    _ancestor_chain,
    _call_escape_verdict,
    _closure_escapes_iteration,
    _comprehension_lambda_findings,
    _comprehension_target_names,
    _enclosing_call,
    _file_closure_findings,
    _loop_closure_findings,
    _loop_def_finding,
    _loop_target_names,
    _loop_target_names_from_target,
    _names_referenced,
    scan_late_binding_closures,
)

from py_ci_shared.code_audit import closures as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
