"""(internal) source_text_assertions now lives in py_ci_shared.code_audit.source_text_assertions; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.source_text_assertions import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CHECK,
    _READ_SPELLINGS,
    _how,
    scan_source_text_assertions,
)

from py_ci_shared.code_audit import source_text_assertions as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
