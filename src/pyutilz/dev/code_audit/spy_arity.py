"""(internal) spy_arity now lives in py_ci_shared.code_audit.spy_arity; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.spy_arity import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _call_positional_count,
    _collect_prod_call_max,
    _find_spy_findings_in_test_file,
    _is_patch_call,
    _positional_arity,
    _real_max_for,
    _resolve_call_qualified,
    _short_name,
    scan_stale_test_spy_arity,
)

from py_ci_shared.code_audit import spy_arity as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
