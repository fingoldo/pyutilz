"""(internal) console_unicode now lives in py_ci_shared.code_audit.console_unicode; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.console_unicode import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _LOG_METHOD_NAMES,
    _UNREADABLE_INIT_WARNED,
    _ancestor_package_has_stdio_utf8_reconfigure,
    _has_non_ascii,
    _has_stdio_utf8_reconfigure,
    _is_console_call,
    _read_init_source,
    _str_arg_values,
    logger,
    scan_console_unicode,
)

from py_ci_shared.code_audit import console_unicode as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
