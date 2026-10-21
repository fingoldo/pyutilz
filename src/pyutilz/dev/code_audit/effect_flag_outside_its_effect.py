"""(internal) effect_flag_outside_its_effect now lives in py_ci_shared.code_audit.effect_flag_outside_its_effect; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.effect_flag_outside_its_effect import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _EXIT_STMTS,
    _LOGGING_RECEIVERS,
    _RECORD_VALUES,
    _VACUOUS_TOKENS,
    _blocks,
    _calls_in,
    _effect_finding,
    _first_unguarded_record,
    _function_findings,
    _has_exit,
    _is_a_success_record,
    _is_exempt_guard,
    _only_initialises,
    _only_logs,
    _shared_tokens,
    _tests_membership_of,
    _unguarded_record,
    scan_effect_flag_outside_its_effect,
)

from py_ci_shared.code_audit import effect_flag_outside_its_effect as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
