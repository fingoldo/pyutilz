"""(internal) comment_names_missing_symbol now lives in py_ci_shared.code_audit.comment_names_missing_symbol; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.comment_names_missing_symbol import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BACKTICKED_CALL,
    _COMMENT_TEXTS_CACHE,
    _COMMENT_TEXTS_CACHE_MAX_ENTRIES,
    _COMMENT_TEXTS_LOCK,
    _LINE_CITATION,
    _cached_comment_texts,
    _citation_line,
    _comment_texts,
    _defined_names,
    logger,
    scan_comment_cites_absolute_line,
    scan_comment_names_missing_symbol,
)

from py_ci_shared.code_audit import comment_names_missing_symbol as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
