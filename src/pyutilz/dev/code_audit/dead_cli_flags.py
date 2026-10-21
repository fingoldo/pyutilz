"""(internal) dead_cli_flags now lives in py_ci_shared.code_audit.dead_cli_flags; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.dead_cli_flags import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _attribute_corpus,
    _dead_flag_name,
    _flag_dest_name,
    _is_add_argument_call,
    _is_self_handling_action,
    logger,
    scan_dead_cli_flags,
)

from py_ci_shared.code_audit import dead_cli_flags as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
