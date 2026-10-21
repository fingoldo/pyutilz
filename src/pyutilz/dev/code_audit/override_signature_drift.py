"""(internal) override_signature_drift now lives in py_ci_shared.code_audit.override_signature_drift; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.override_signature_drift import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _MIN_SIBLINGS,
    _accepts_anything,
    _ancestors,
    _base_names,
    _collect_widened_params,
    _declared_params,
    _decorator_names,
    _index_classes,
    _is_stub,
    _methods,
    _placeholder_variants,
    scan_override_signature_drift,
)

from py_ci_shared.code_audit import override_signature_drift as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
