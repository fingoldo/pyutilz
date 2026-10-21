"""(internal) duplicate_credential_regex now lives in py_ci_shared.code_audit.duplicate_credential_regex; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.duplicate_credential_regex import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    DEFAULT_CREDENTIAL_KEYWORDS_RE,
    scan_duplicate_credential_regex,
)

from py_ci_shared.code_audit import duplicate_credential_regex as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
