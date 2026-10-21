"""(internal) docstring_args now lives in py_ci_shared.code_audit.docstring_args; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.docstring_args import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _ARGS_HEADER_RE,
    _ARG_NAME_RE,
    _IGNORED_PARAM_NAMES,
    _NEXT_SECTION_RE,
    _documented_arg_names,
    scan_docstring_args_completeness,
)

from py_ci_shared.code_audit import docstring_args as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
