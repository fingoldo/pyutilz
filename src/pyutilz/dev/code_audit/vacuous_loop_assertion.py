"""(internal) vacuous_loop_assertion now lives in py_ci_shared.code_audit.vacuous_loop_assertion; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.vacuous_loop_assertion import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ASSERT_CALLS,
    _assertion_nodes,
    _cannot_be_empty,
    _is_nonempty_literal,
    _nonempty_literal_names,
    _vacuous_loop_findings,
    scan_vacuous_loop_assertion,
)

from py_ci_shared.code_audit import vacuous_loop_assertion as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
