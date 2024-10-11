# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

# ensure_installed("numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numpy as np


def get_topk_indices(arr: np.ndarray, k: int = 1, axis: int = -1, highest: bool = True) -> np.ndarray:
    """Return indices of top-k highest or lowest elements of an array.
    >>>arr=np.array([2., 0., 3.], dtype=float32)
    >>>get_topk_indices(arr, k=2, highest=True)
    array([2, 0], dtype=int64)
    >>>get_topk_indices(arr, k=2, highest=False)
    array([1, 0], dtype=int64))
    """
    if highest:
        indices = np.argpartition(arr, -k, axis=axis)[-k:]
    else:
        indices = np.argpartition(arr, k, axis=axis)[:k]

    indices = indices[np.argsort(arr[indices], axis=axis)]

    if highest:
        return indices[::-1]
    else:
        return indices


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
