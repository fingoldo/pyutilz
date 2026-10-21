"""(internal) near_duplicate_function_body now lives in py_ci_shared.code_audit.near_duplicate_function_body; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.near_duplicate_function_body import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BUILTIN_CALL_NAMES,
    _DEFAULT_MIN_CONTAINMENT,
    _DEFAULT_MIN_SIMILARITY,
    _MAX_POSTING_LIST,
    _MAX_TOKENS_FOR_EXACT_CHECK,
    _MIN_BODY_NODES,
    _MIN_SHARED_SHINGLE_FRACTION,
    _SHINGLE_SIZE,
    _TOKEN_RE,
    _called_names,
    _candidate_pairs,
    _collect_entries,
    _delegates_to_shared_helper,
    _is_deprecated_alias_boilerplate,
    _is_nested,
    _near_duplicate_finding,
    _pair_finding,
    _shingles,
    _subset_finding,
    _tokenize,
    scan_near_duplicate_function_body,
)

from py_ci_shared.code_audit import near_duplicate_function_body as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
