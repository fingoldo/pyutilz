# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed

# ensure_installed("numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import numpy as np


def get_topk_indices(arr: np.ndarray, k: int = 1, axis: int = -1, highest: bool = True) -> np.ndarray:
    """Return indices of top-k highest or lowest elements along a given axis.

    Works for arrays of any dimensionality. The result has k entries along `axis`,
    ordered from best to worst (highest-first when highest=True, lowest-first otherwise).

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

    if highest:
        # Partition so the k largest end up in the last k positions along axis.
        part = np.argpartition(arr, n - k, axis=axis)
        cand = np.take(part, np.arange(n - k, n), axis=axis)
    else:
        part = np.argpartition(arr, k - 1, axis=axis)
        cand = np.take(part, np.arange(0, k), axis=axis)

    # Reorder the k candidates by their actual values (argpartition leaves them unordered).
    cand_vals = np.take_along_axis(arr, cand, axis=axis)
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
    """Returns (a-b)/b,
    but watches that b is not close to zero by shifting both values up, so that b.min() becomes positive and at least as big as the entire span of a or b.
    span is maybe 99% percentile span.
    """

    return div0(a - b, b + span_correction, na_fill=na_fill)
