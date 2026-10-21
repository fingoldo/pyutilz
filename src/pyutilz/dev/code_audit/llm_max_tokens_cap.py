"""(internal) llm_max_tokens_cap now lives in py_ci_shared.code_audit.llm_max_tokens_cap; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.llm_max_tokens_cap import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _CAPPED_METHOD_NAMES,
    _is_zero_literal,
    _provider_var_names,
    scan_llm_call_missing_max_tokens_cap,
)

from py_ci_shared.code_audit import llm_max_tokens_cap as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
