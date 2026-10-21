"""(internal) timing_literal_in_call now lives in py_ci_shared.code_audit.timing_literal_in_call; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.timing_literal_in_call import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BACKOFF_KEYWORDS,
    _SKIPPED_TOP_DIRS,
    _SLEEP_CALLS,
    _call_name,
    _nonzero_numbers,
    _timing_literals_in_calls,
    scan_timing_literal_in_call,
)

from py_ci_shared.code_audit import timing_literal_in_call as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
