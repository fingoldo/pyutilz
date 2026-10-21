"""(internal) return_annotation now lives in py_ci_shared.code_audit.return_annotation; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.return_annotation import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CONTAINER_LITERAL_TYPES,
    _OwnReturnFinder,
    _SCALAR_ANNOTATION_NAMES,
    _annotation_name,
    _mismatch_detail,
    scan_return_annotation_mismatch,
)

from py_ci_shared.code_audit import return_annotation as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
