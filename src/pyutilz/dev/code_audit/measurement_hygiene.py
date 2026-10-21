"""(internal) measurement_hygiene now lives in py_ci_shared.code_audit.measurement_hygiene; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.measurement_hygiene import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BARE_DIGITS_RE,
    _GROUPED_INT,
    _NUMBER_CLAIM_RE,
    _assert_lower_bounds,
    _documented_claim,
    _int_or_float_conversion_nearby,
    scan_regex_integer_parse,
    scan_thresholds_below_documented_result,
)

from py_ci_shared.code_audit import measurement_hygiene as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
