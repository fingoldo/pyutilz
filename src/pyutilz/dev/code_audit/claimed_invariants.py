"""(internal) claimed_invariants now lives in py_ci_shared.code_audit.claimed_invariants; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.claimed_invariants import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CLAIM_RE,
    _callers_within,
    _claim_sentence,
    _is_unenforced_claim,
    _test_corpus,
    scan_unenforced_docstring_invariants,
)

from py_ci_shared.code_audit import claimed_invariants as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
