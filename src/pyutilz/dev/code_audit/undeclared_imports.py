"""(internal) undeclared_imports now lives in py_ci_shared.code_audit.undeclared_imports; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.undeclared_imports import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _DOMAIN_OWN_GROUP,
    _PACKAGE_OWN_GROUPS,
    _STDLIB,
    _domain_for,
    _top_level_import_roots,
    scan_undeclared_imports,
)

from py_ci_shared.code_audit import undeclared_imports as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
