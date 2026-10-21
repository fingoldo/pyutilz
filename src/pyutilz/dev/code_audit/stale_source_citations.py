"""(internal) stale_source_citations now lives in py_ci_shared.code_audit.stale_source_citations; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.stale_source_citations import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CITATION,
    _citation_problem,
    _comment_and_docstring_tokens,
    _docstring_lines,
    _index_tree,
    _resolve,
    logger,
    scan_stale_source_citations,
)

from py_ci_shared.code_audit import stale_source_citations as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
