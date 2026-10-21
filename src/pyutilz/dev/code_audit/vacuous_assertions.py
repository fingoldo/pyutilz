"""(internal) vacuous_assertions now lives in py_ci_shared.code_audit.vacuous_assertions; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.vacuous_assertions import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _collect_top_level_asserts,
    _is_bare_is_not_none,
    _is_empty_literal,
    _looks_like_full_domain_or,
    _vacuous_operand_target,
    scan_tautological_is_not_none_only_tests,
    scan_vacuous_assertions,
)

from py_ci_shared.code_audit import vacuous_assertions as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
