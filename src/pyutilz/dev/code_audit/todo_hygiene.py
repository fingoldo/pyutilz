"""(internal) todo_hygiene now lives in py_ci_shared.code_audit.todo_hygiene; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.todo_hygiene import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    scan_todo_hygiene,
)

from py_ci_shared.code_audit import todo_hygiene as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
