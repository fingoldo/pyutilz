"""(internal) vacuous_matching now lives in py_ci_shared.code_audit.vacuous_matching; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.vacuous_matching import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _GUARD_TEMPLATES,
    _bare_name_iterable_of_all,
    _is_guarded,
    scan_vacuous_empty_pattern_match,
)

from py_ci_shared.code_audit import vacuous_matching as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
