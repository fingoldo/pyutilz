"""(internal) credential_logging now lives in py_ci_shared.code_audit.credential_logging; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.credential_logging import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CREDENTIAL_NAME_RE,
    _LOG_METHODS,
    _REDACT_HINT_RE,
    _is_logger_receiver,
    _looks_credential_shaped,
    scan_credential_shaped_log_args,
)

from py_ci_shared.code_audit import credential_logging as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
