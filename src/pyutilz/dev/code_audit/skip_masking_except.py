"""(internal) skip_masking_except now lives in py_ci_shared.code_audit.skip_masking_except; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.skip_masking_except import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _body_is_imports_only,
    _except_body_calls_pytest_skip,
    scan_except_skip_masks_call_under_test,
)

from py_ci_shared.code_audit import skip_masking_except as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
