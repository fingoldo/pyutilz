"""(internal) duplicate_conditions now lives in py_ci_shared.code_audit.duplicate_conditions; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.duplicate_conditions import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _boolop_findings,
    _contains_bare_function_call,
    _dict_key_findings,
    _elif_branch_ids,
    _elif_chain_findings,
    _operand_key,
    scan_duplicate_conditions,
)

from py_ci_shared.code_audit import duplicate_conditions as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
