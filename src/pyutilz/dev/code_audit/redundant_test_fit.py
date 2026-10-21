"""(internal) redundant_test_fit now lives in py_ci_shared.code_audit.redundant_test_fit; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.redundant_test_fit import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CHEAP_CALLEES,
    _call_signature,
    _duplicate_call_findings,
    _enclosing_test_functions,
    _is_literal_data_factory,
    _iteration_bound_names,
    _literal_call_occurrences,
    logger,
    scan_redundant_test_fit_calls,
)

from py_ci_shared.code_audit import redundant_test_fit as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
