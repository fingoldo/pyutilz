"""(internal) mojibake now lives in py_ci_shared.code_audit.mojibake; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.mojibake import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _MIN_RUN_LENGTH,
    _NON_ASCII_RUN_RE,
    _is_roundtrip_mojibake,
    logger,
    scan_mojibake,
)

from py_ci_shared.code_audit import mojibake as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
