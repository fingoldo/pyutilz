"""(internal) unraised_exceptions now lives in py_ci_shared.code_audit.unraised_exceptions; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.unraised_exceptions import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _KNOWN_EXCEPTION_BASES,
    _collect_class_defs,
    _collect_exception_classes,
    _collect_raised_names,
    _covered_via_subclass,
    _exception_base_names,
    _parse_trees,
    _resolve_exception_class_names,
    scan_unraised_exceptions,
)

from py_ci_shared.code_audit import unraised_exceptions as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
