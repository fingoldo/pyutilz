"""(internal) provenance_flow now lives in py_ci_shared.code_audit.provenance_flow; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.provenance_flow import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _MIN_KEY_LEN,
    _collect,
    _is_field_like,
    _note_read_site,
    _note_site,
    _note_write_site,
    scan_record_field_flow,
)

from py_ci_shared.code_audit import provenance_flow as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
