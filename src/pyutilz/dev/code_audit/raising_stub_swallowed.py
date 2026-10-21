"""(internal) raising_stub_swallowed now lives in py_ci_shared.code_audit.raising_stub_swallowed; this module aliases it so the old path keeps working."""

import sys as _sys

from py_ci_shared.code_audit.raising_stub_swallowed import (  # noqa: F401 - re-exported so the names stay visible to type checkers
    _BROAD,
    _PATCH_FUNCS,
    _asserts_on_a_raise,
    _handler_is_broad,
    _installs_raiser,
    _partition_swallowing_production,
    _patched_attr_of,
    _patched_targets_of_raising_stubs,
    _raises_unconditionally,
    _reraises,
    _spy_style_assertions,
    _stub_names_that_raise,
    _stub_swallow_findings,
    _swallowing_calls,
    scan_raising_stub_swallowed,
)

from py_ci_shared.code_audit import raising_stub_swallowed as _impl

# Runtime identity: the old path IS the moved module, so a test that rebinds a module constant reaches the code that reads it.
_sys.modules[__name__] = _impl
