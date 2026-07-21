"""Numpy-based array/vector helper utilities."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------


# ensure_installed("numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np


def get_topk_indices(arr: np.ndarray, k: int = 1, axis: int = -1, highest: bool = True) -> np.ndarray:
    """Return indices of top-k highest or lowest elements along a given axis.

    Works for arrays of any dimensionality. The result has k entries along `axis`,
    ordered from best to worst (highest-first when highest=True, lowest-first otherwise).

    NaN is never picked as a "highest" value (nor excluded from being "lowest" when there aren't
    enough real values to fill k slots) -- ``np.argpartition``/``np.argsort`` otherwise treat NaN
    as greater than every real number, so an unguarded NaN anywhere in ``arr`` would silently
    outrank the true maximum.

    >>> arr = np.array([2., 0., 3.], dtype=np.float32)
    >>> get_topk_indices(arr, k=2, highest=True)
    array([2, 0], dtype=int64)
    >>> get_topk_indices(arr, k=2, highest=False)
    array([1, 0], dtype=int64)
    """
    arr = np.asarray(arr)
    n = arr.shape[axis]
    if k > n:
        raise ValueError(f"k={k} exceeds array length {n} along axis {axis}")

    if k == 0:
        empty_shape = list(arr.shape)
        empty_shape[axis] = 0
        return np.empty(empty_shape, dtype=np.int64)

    # NaN sorts as +inf under argpartition/argsort's default order; substitute it towards the
    # "worst" end for whichever direction we're ranking so it's never mistaken for a real extreme
    # (only for floating dtypes -- np.isnan raises on integer arrays, which can't hold NaN anyway).
    is_float = np.issubdtype(arr.dtype, np.floating)
    if highest:
        ranking_arr = np.where(np.isnan(arr), -np.inf, arr) if is_float else arr
        # Partition so the k largest end up in the last k positions along axis.
        part = np.argpartition(ranking_arr, n - k, axis=axis)
        cand = np.take(part, np.arange(n - k, n), axis=axis)
    else:
        ranking_arr = np.where(np.isnan(arr), np.inf, arr) if is_float else arr
        part = np.argpartition(ranking_arr, k - 1, axis=axis)
        cand = np.take(part, np.arange(0, k), axis=axis)

    # Reorder the k candidates by their (NaN-substituted) ranking values (argpartition leaves
    # them unordered).
    cand_vals = np.take_along_axis(ranking_arr, cand, axis=axis)
    order = np.argsort(cand_vals, axis=axis)
    if highest:
        order = np.flip(order, axis=axis)
    return np.take_along_axis(cand, order, axis=axis)


def div0(a, b, na_fill=np.nan):
    """a / b, divide by 0 -> `fill`
    div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
    div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
    if np.isscalar(c):
        return c if np.isfinite(c) else na_fill
    else:
        c[~np.isfinite(c)] = na_fill
        return c


def smart_ratios(a: np.ndarray, b: np.ndarray, span_correction: float = 0.0, na_fill=np.nan) -> np.ndarray:
    """Returns (a-b)/(b + span_correction), guarding only against an exactly-zero or non-finite
    denominator (via :func:`div0`).

    ``span_correction`` is NOT computed automatically from the data -- it's a caller-supplied
    constant added to ``b`` only (default 0.0, i.e. no shift). If ``b`` contains values merely
    close to (but not exactly) zero, the ratio can still blow up; pass an explicit
    ``span_correction`` (e.g. derived from ``np.abs(b).max() - b.min()`` or a percentile span of
    ``a``/``b``) to dampen that, matching whatever numeric-stability margin your use case needs.
    """

    return div0(a - b, b + span_correction, na_fill=na_fill)  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
