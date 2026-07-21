"""Meta-test proposed after the 2026-07-21 audit round 2 (MEDIUM finding, API design consistency):
the dataframe-cleanup function family (pandaslib.dtypes/frames, polarslib) originally had a mix of
``verbose: bool`` and ``verbose: int`` annotations for a parameter that every one of these
functions only ever consults via ``if verbose:`` truthiness -- never graded (0/1/2) logging detail.
The int annotation misleads a caller into believing ``verbose=2`` requests more detail than
``verbose=1``, and porting a call from one sibling function to another with a mismatched annotation
silently "worked" only because Python doesn't enforce the hint at runtime.

Fixed by standardizing every one of these on ``verbose: bool``. Since Python does NOT enforce type
annotations, no ordinary call-based test would catch a regression back to ``int`` -- this test pins
the annotation itself via ``inspect.signature`` so a future edit reverting the type is caught
immediately, without needing to touch pytest.mark.parametrize'd behavioral tests at all.

Deliberately scoped to this specific "truthiness-only verbose toggle" family, not every ``verbose``
parameter in the package -- ``dev/benchmarking.py`` and ``system/parallel.py`` (joblib's own
``verbose=0..50`` convention) genuinely use graded int verbosity levels and are correctly typed
``int`` on purpose.
"""
from __future__ import annotations

import inspect

import pyutilz.data.pandaslib.dtypes as dtypes
import pyutilz.data.pandaslib.frames as frames
import pyutilz.data.polarslib as polarslib

# (module, function_name) pairs whose "verbose" parameter must be annotated exactly `bool`.
_TRUTHINESS_ONLY_VERBOSE_FUNCTIONS = [
    (dtypes, "optimize_dtypes"),
    (dtypes, "ensure_dataframe_float32_convertability"),
    (frames, "nullify_standard_values"),
    (frames, "remove_constant_columns"),
    (polarslib, "bin_numerical_columns"),
    (polarslib, "drop_constant_columns"),
]


def test_truthiness_only_verbose_functions_still_exist():
    """A rename on any pinned function without updating this list is exactly the drift this test
    guards against -- fail loudly rather than silently stop checking a renamed function."""
    missing = []
    for module, func_name in _TRUTHINESS_ONLY_VERBOSE_FUNCTIONS:
        if not hasattr(module, func_name):
            missing.append(f"{module.__name__}.{func_name}")
    assert not missing, f"pinned function(s) no longer exist -- update _TRUTHINESS_ONLY_VERBOSE_FUNCTIONS or restore them: {missing}"


def test_verbose_parameter_is_annotated_bool_not_int():
    mismatches = []
    for module, func_name in _TRUTHINESS_ONLY_VERBOSE_FUNCTIONS:
        fn = getattr(module, func_name)
        sig = inspect.signature(fn)
        param = sig.parameters.get("verbose")
        if param is None:
            mismatches.append(f"{module.__name__}.{func_name} no longer has a 'verbose' parameter at all")
            continue
        if param.annotation is not bool:
            mismatches.append(f"{module.__name__}.{func_name}(verbose: {param.annotation!r} = {param.default!r}) -- expected `verbose: bool`")
    assert not mismatches, (
        "verbose parameter drifted away from `bool` on a function that only ever checks it via "
        "truthiness -- either restore `verbose: bool`, or if graded int verbosity is now genuinely "
        "implemented and documented, remove that function from _TRUTHINESS_ONLY_VERBOSE_FUNCTIONS:\n  " + "\n  ".join(mismatches)
    )
