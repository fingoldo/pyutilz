"""(internal) locals_globals_output now lives in py_ci_shared.code_audit.locals_globals_output; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.locals_globals_output import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _OUTPUT_LIKE_KWARG_NAMES,
    _READ_ONLY_BUILTIN_CONSUMERS,
    _kwarg_output_findings,
    scan_locals_globals_as_output,
)

from py_ci_shared.code_audit import locals_globals_output as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
