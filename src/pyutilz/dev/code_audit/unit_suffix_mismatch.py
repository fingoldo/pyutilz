"""(internal) unit_suffix_mismatch now lives in py_ci_shared.code_audit.unit_suffix_mismatch; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.unit_suffix_mismatch import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _SYNONYMS,
    _UNIT_FAMILIES,
    _assigned_pairs,
    _source_name,
    _subscript_key,
    _target_names,
    _unit_of,
    scan_unit_suffix_mismatch,
)

from py_ci_shared.code_audit import unit_suffix_mismatch as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
