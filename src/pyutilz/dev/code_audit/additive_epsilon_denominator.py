"""(internal) additive_epsilon_denominator now lives in py_ci_shared.code_audit.additive_epsilon_denominator; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.additive_epsilon_denominator import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_EPSILON_MAX,
    _epsilon_padded_names,
    _is_epsilon_constant,
    _is_epsilon_sum,
    scan_additive_epsilon_denominator,
)

from py_ci_shared.code_audit import additive_epsilon_denominator as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
