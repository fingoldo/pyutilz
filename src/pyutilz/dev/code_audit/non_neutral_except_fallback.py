"""(internal) non_neutral_except_fallback now lives in py_ci_shared.code_audit.non_neutral_except_fallback; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.non_neutral_except_fallback import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_AUDIBLE_FUNCTIONS,
    DEFAULT_AUDIBLE_LOG_METHODS,
    _DETERMINATE_NEGATIVE_EXCEPTIONS,
    _MAX_TOTAL_FUNCTION_GUARD_STMTS,
    _describe_literal,
    _flagged_substitution,
    _guards_an_import,
    _handler_exception_names,
    _is_audible,
    _is_determinate_negative_only,
    _is_import_error_only,
    _is_single_expression_guard,
    _node_substitutions,
    _own_nodes,
    _substituted_literal,
    _target_name,
    scan_non_neutral_except_fallback,
)

from py_ci_shared.code_audit import non_neutral_except_fallback as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
