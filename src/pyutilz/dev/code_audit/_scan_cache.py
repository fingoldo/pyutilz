"""(internal) _scan_cache now lives in py_ci_shared.code_audit._scan_cache; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit._scan_cache import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _FINDING_FIELDS,
    _PACKAGE_DIR,
    _as_json,
    _cache_dir,
    _signature,
    cache_key,
    load,
    logger,
    store,
)

from py_ci_shared.code_audit import _scan_cache as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
