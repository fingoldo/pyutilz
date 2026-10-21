"""(internal) domain_boundary now lives in py_ci_shared.code_audit.domain_boundary; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.domain_boundary import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    BoundarySymbol,
    _iter_named,
    _span,
    _term_pattern,
    scan_domain_vocabulary_leak,
)

from py_ci_shared.code_audit import domain_boundary as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
