"""(internal) nondiscriminating_test now lives in py_ci_shared.code_audit.nondiscriminating_test; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.nondiscriminating_test import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_ASSERTING_CALLS,
    DEFAULT_TEST_PREFIX,
    _FUNC_NODES,
    _REASON_HELP,
    _every_assert_is_conditional,
    _has_any_check,
    _has_imperative_xfail,
    _has_pass_only_if,
    _own_nodes,
    _reasons,
    _swallows_assertion_error,
    scan_nondiscriminating_test_functions,
)

from py_ci_shared.code_audit import nondiscriminating_test as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
