"""(internal) mutation_during_iteration now lives in py_ci_shared.code_audit.mutation_during_iteration; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.mutation_during_iteration import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _MUTATING_DICT_METHODS,
    _MUTATING_LIST_METHODS,
    _MUTATING_SET_METHODS,
    _assign_findings,
    _call_findings,
    _delete_findings,
    _iter_target_chain,
    _looks_dict_like,
    _loop_mutation_findings,
    scan_mutation_during_iteration,
)

from py_ci_shared.code_audit import mutation_during_iteration as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
