"""(internal) stats_key_coverage now lives in py_ci_shared.code_audit.stats_key_coverage; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.stats_key_coverage import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _COUNTER_ATTRS,
    _INC_HELPERS,
    _declared_by,
    _dict_literal_keys,
    _is_counter_attribute,
    _note_key,
    _record_class,
    _undeclared_key_finding,
    _written_by_augassign,
    _written_by_call,
    _written_by_read_modify_write,
    _written_keys,
    scan_stats_key_coverage,
)

from py_ci_shared.code_audit import stats_key_coverage as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
