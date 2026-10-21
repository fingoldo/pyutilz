"""(internal) settings_container_nodecode now lives in py_ci_shared.code_audit.settings_container_nodecode; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.settings_container_nodecode import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CONTAINERS,
    _before_validated_fields,
    _decoding_disabled,
    _has_nodecode,
    _is_container,
    _is_true,
    _keyword,
    _name_of,
    _settings_classes,
    _split_annotated,
    scan_settings_container_field_needs_nodecode,
)

from py_ci_shared.code_audit import settings_container_nodecode as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
