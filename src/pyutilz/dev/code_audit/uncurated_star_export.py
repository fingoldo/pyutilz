"""(internal) uncurated_star_export now lives in py_ci_shared.code_audit.uncurated_star_export; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.uncurated_star_export import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _module_defines_all,
    _resolve_relative_module_path,
    _star_import_targets,
    scan_uncurated_star_exports,
)

from py_ci_shared.code_audit import uncurated_star_export as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
