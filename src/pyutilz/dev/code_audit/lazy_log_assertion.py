"""(internal) lazy_log_assertion now lives in py_ci_shared.code_audit.lazy_log_assertion; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.lazy_log_assertion import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CALL_RECORD_ATTRS,
    _HAS_DIGIT,
    _LOG_LEVELS,
    _UNITTEST_ASSERTIONS,
    _asserted_literals,
    _assertion_expression,
    _could_be_an_eager_rendering,
    _format_strings,
    _is_unrenderable_literal,
    _touches_a_log_record,
    scan_lazy_log_assertion,
)

from py_ci_shared.code_audit import lazy_log_assertion as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
