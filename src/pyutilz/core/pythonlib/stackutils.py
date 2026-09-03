"""Call-stack introspection: caller lookups, parameter capture/restore, soft field access.

Split out of the historical flat ``pyutilz.core.pythonlib`` module; re-exported from the
package ``__init__`` to preserve the public import surface.
"""

import sys

from ._common import Any, Optional, Sequence, inspect, logger

# ----------------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------------


def get_or_warn(obj: dict, field: str, target: str) -> Optional[Any]:
    """Returns `obj[field]` if present, else logs a warning naming `target` and returns None."""
    desired = obj.get(field)
    if desired is None:
        logger.warning("No %s field in %s %s", field, target, obj)
    return desired


# ----------------------------------------------------------------------------------------------------------------------------
# Stack
# ----------------------------------------------------------------------------------------------------------------------------


def lookup_in_stack(variable):
    """Searches the call stack (innermost frame first) for a global named `variable` and returns its value.

    Returns None if no calling frame's globals define a truthy value for it.
    """
    # Walk ``f_back`` rather than calling ``inspect.stack()``: the latter materialises a
    # ``FrameInfo`` for EVERY frame (reading and caching each one's source context) before the
    # first candidate is even examined, and the returned list pins every enclosing frame's locals
    # -- arrays included -- for as long as the caller holds it. The loop below touches only
    # ``f_globals`` and drops its frame reference in the ``finally``.
    frame: Optional[Any] = sys._getframe(1)
    try:
        while frame is not None:
            res = frame.f_globals.get(variable)
            if res:
                return res
            frame = frame.f_back
    finally:
        del frame
    return None


def get_parent_func_args(skip_args: Sequence = ("self",)) -> dict:
    """Get arg-values of a caller func as a dict."""

    _this_frame = inspect.currentframe()
    previous_frame = _this_frame.f_back if _this_frame is not None else None
    if previous_frame is None:
        return {}
    args_info = inspect.getargvalues(previous_frame)

    # Materialise the frame's locals into a plain dict ONCE rather than calling
    # ``args_info.locals.get(arg)`` per parameter name: on 3.13+, ``args_info.locals`` is a
    # ``FrameLocalsProxy`` whose ``.get()`` is measurably slower than a native dict's per call
    # (cProfile on a real ~300-parameter caller -- mlframe's ``MRMR.__init__`` -- shows 236500
    # ``FrameLocalsProxy.get()`` calls costing 0.191s tottime vs 0.050s for the same call count
    # against a materialised plain dict, ~3.8x per call; end-to-end this call + the sibling
    # ``store_params_in_object`` drop from 0.691s to 0.366s cumulative over 500 calls under
    # cProfile, ~1.9x). Wall-clock A/B on a loaded dev box was too noisy to pin an absolute
    # number reliably, but the cProfile relative comparison (same instrumentation on both sides,
    # so its overhead cancels out) is a controlled, repeatable measurement, not a fluke. A real,
    # non-negligible cost when constructing MANY instances (sklearn.clone() / bootstrap
    # replicates / GridSearchCV candidates). Same filtering as before (only real parameter
    # names, never incidental locals): the ``for arg in args_info.args`` loop below is
    # unchanged, so this is a pure speed win, not a behavior change.
    _locals_dict = dict(args_info.locals)
    argvals = {arg: _locals_dict.get(arg) for arg in args_info.args if arg not in skip_args}
    return argvals


def store_params_in_object(obj: object, params: dict, postfix: str = "_param_"):
    """Useful for persisting __init__ params in the class instance.

    ``postfix`` defaults to ``"_param_"`` to round-trip with :func:`load_object_params_into_func`'s
    own default of the same value -- the two are documented as an inverse pair and must agree on
    the suffix, or the round trip silently loses every value.
    """
    if obj is None:
        return
    for param_name, param_value in params.items():
        setattr(obj, param_name + postfix, param_value)


def load_object_params_into_func(obj: object, locals: dict, postfix: str = "_param_"):  # noqa: A002 -- public API (pyutilz.__init__ alias), signature tracked by tests/test_meta/test_api_stability.py
    """Inverse of :func:`store_params_in_object`: collect ``obj``'s ``*postfix`` attributes.

    The collected ``{name: value}`` mapping is written into ``locals`` AND returned. Passing a
    function's own ``locals()`` does NOT restore anything: writes to an optimized frame's
    ``locals()`` snapshot never reach its fast locals, in every CPython version, so the target
    parameters stayed ``None`` and nothing signalled it. Use the RETURN value instead::

        params = load_object_params_into_func(obj, {})
        alpha = params.get("alpha", alpha)

    Returns:
        dict: the ``{stripped_attr_name: value}`` pairs found on ``obj`` (empty when ``obj`` is None).
    """
    collected: dict = {}
    if obj is None:
        return collected
    for attr in dir(obj):
        if attr.endswith(postfix):
            key, value = attr[: -len(postfix)], getattr(obj, attr)
            collected[key] = value
            locals[key] = value
    return collected
