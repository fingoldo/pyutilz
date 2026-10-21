"""(internal) sql_migrations now lives in py_ci_shared.code_audit.sql_migrations; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.sql_migrations import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BLOCK_COMMENT_RE,
    _DO_BLOCK_RE,
    _DO_BLOCK_SPAN_RE,
    _EXISTENCE_GUARD_RE,
    _LINE_COMMENT_RE,
    _UNGUARDED_ADD_PK_RE,
    _UNGUARDED_PATTERNS,
    _existence_guarded_line_ranges,
    _strip_sql_comments,
    logger,
    scan_sql_migration_idempotency,
)

from py_ci_shared.code_audit import sql_migrations as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
