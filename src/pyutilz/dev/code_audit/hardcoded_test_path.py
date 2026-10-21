"""(internal) hardcoded_test_path now lives in py_ci_shared.code_audit.hardcoded_test_path; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.hardcoded_test_path import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ABS_PATH_RE,
    scan_hardcoded_absolute_path_in_test,
)

from py_ci_shared.code_audit import hardcoded_test_path as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
