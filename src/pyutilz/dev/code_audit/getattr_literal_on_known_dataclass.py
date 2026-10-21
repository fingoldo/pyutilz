"""(internal) getattr_literal_on_known_dataclass now lives in py_ci_shared.code_audit.getattr_literal_on_known_dataclass; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.getattr_literal_on_known_dataclass import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _assigned_dataclass_types,
    _attribute_store_names,
    _class_shapes,
    _dataclass_fields,
    _getattr_literal_findings,
    _is_dataclass_decorator,
    _local_dataclass_types,
    _resolved_members,
    scan_getattr_literal_on_known_dataclass,
)

from py_ci_shared.code_audit import getattr_literal_on_known_dataclass as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
