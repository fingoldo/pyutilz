"""(internal) test_asserts_against_production_constant now lives in py_ci_shared.code_audit.test_asserts_against_production_constant; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.test_asserts_against_production_constant import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ASSERT_HELPERS,
    _PATH_NAME_SUFFIXES,
    _UNITTEST_ASSERT_HELPERS,
    _arithmetic_over,
    _asserted_comparisons,
    _imported_from_production,
    _is_path_construction,
    _mentions,
    _unwrap_approx,
    scan_test_asserts_against_production_constant,
)

from py_ci_shared.code_audit import test_asserts_against_production_constant as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
