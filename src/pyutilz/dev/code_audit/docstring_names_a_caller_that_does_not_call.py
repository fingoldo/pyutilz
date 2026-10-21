"""(internal) docstring_names_a_caller_that_does_not_call now lives in py_ci_shared.code_audit.docstring_names_a_caller_that_does_not_call; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.docstring_names_a_caller_that_does_not_call import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CLAIM,
    _call_graph,
    _calls_made_by,
    _documented_callers,
    _mismatched_caller_findings,
    _parse_tree_files,
    scan_docstring_names_a_caller_that_does_not_call,
)

from py_ci_shared.code_audit import docstring_names_a_caller_that_does_not_call as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
