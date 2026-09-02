"""Fused per-column reductions used to build content hashes of large ndarrays.

Hashing an array by its (shape, dtype, head/tail rows, per-column sum/min/max) summary is only
worth doing if the summary itself is cheap. Computing it as three separate numpy reductions is not:
``arr.sum(axis=0)``, ``arr.min(axis=0)`` and ``arr.max(axis=0)`` make three full passes over the
data, each in the strided (cache-hostile) direction for a C-contiguous array. One fused numba pass
computes all three per column while each cache line is still resident, and parallelises across
column blocks. Measured on a (2_000_000, 4) float64 C-contiguous array: 3-pass numpy 150-154 ms vs
fused njit 4.6-5.5 ms (27-33x).

The numpy path is kept as the fallback for anything the kernel cannot take (numba absent, complex /
float16 / non-numeric dtypes) so behaviour never depends on numba being importable.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["column_sum_min_max", "NUMBA_SUPPORTED_KINDS"]

# dtype kinds the fused kernel accepts: signed int, unsigned int, real float. Complex is excluded
# (numba has no lexicographic complex min/max, numpy does) and so is float16 (no numba support);
# both fall through to the numpy path, which is what they always used.
NUMBA_SUPPORTED_KINDS = frozenset("iuf")

try:
    from numba import njit, prange

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dep of the package today
    _HAVE_NUMBA = False


if _HAVE_NUMBA:

    @njit(cache=True, parallel=True, nogil=True)
    def _fused_col_reductions(a2: Any) -> Tuple[Any, Any, Any]:  # pragma: no cover - compiled by numba, invisible to coverage (NUMBA_DISABLE_JIT=1 restores it)
        """One pass over ``a2`` (2-D) returning per-column ``(sum, min, max)`` as float64 arrays.

        NaN is propagated the way numpy's ``min``/``max`` do -- a column containing any NaN reports
        NaN for all three -- rather than being skipped by the ``<``/``>`` comparisons, which would
        make the summary blind to a NaN that a later numpy build would have surfaced.
        """
        n_rows, n_cols = a2.shape
        sums = np.empty(n_cols, dtype=np.float64)
        mins = np.empty(n_cols, dtype=np.float64)
        maxs = np.empty(n_cols, dtype=np.float64)
        for j in prange(n_cols):  # type: ignore[attr-defined]  # numba types prange as non-iterable; only surfaces where numba ships the stub
            acc = np.float64(0.0)
            lo = np.float64(a2[0, j])
            hi = lo
            saw_nan = lo != lo
            for i in range(n_rows):
                v = np.float64(a2[i, j])
                acc += v
                if v != v:
                    saw_nan = True
                elif v < lo:
                    lo = v
                elif v > hi:
                    hi = v
            if saw_nan:
                lo = np.float64(np.nan)
                hi = np.float64(np.nan)
            sums[j] = acc
            mins[j] = lo
            maxs[j] = hi
        return sums, mins, maxs


def _numpy_col_reductions(arr: np.ndarray, col_axis: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference three-pass numpy implementation: per-column ``(sum, min, max)`` as flat float64."""
    col_sum = np.asarray(arr.sum(axis=col_axis, dtype=np.float64)).ravel()
    col_min = np.asarray(arr.min(axis=col_axis)).astype(np.float64, copy=False).ravel()
    col_max = np.asarray(arr.max(axis=col_axis)).astype(np.float64, copy=False).ravel()
    return col_sum, col_min, col_max


def column_sum_min_max(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-LAST-AXIS-column ``(sum, min, max)`` of an ndarray with ``ndim >= 2``, as float64 arrays.

    Reduces over every axis except the last, i.e. the same ``axis=tuple(range(ndim - 1))`` the numpy
    three-pass version used, so the outputs are element-for-element comparable. Uses the fused numba
    kernel where the dtype allows and silently falls back to numpy otherwise.
    """
    if arr.ndim < 2:
        raise ValueError(f"column_sum_min_max expects ndim >= 2, got {arr.ndim}")
    col_axis = tuple(range(arr.ndim - 1))
    if not _HAVE_NUMBA or arr.dtype.kind not in NUMBA_SUPPORTED_KINDS or arr.dtype == np.float16:
        return _numpy_col_reductions(arr, col_axis)
    try:
        # ``reshape(-1, n_cols)`` is a VIEW for any array whose last axis is the fastest-varying one
        # (the overwhelmingly common case); it copies only for genuinely awkward strides, which is
        # still cheaper than the three strided passes it replaces.
        a2 = arr.reshape(-1, arr.shape[-1])
        sums, mins, maxs = _fused_col_reductions(a2)
        return sums, mins, maxs
    except Exception as e:
        # A dtype/layout numba rejects at compile time must degrade to the numpy answer, never break
        # the caller's hash.
        logger.debug("Fused column reduction unavailable (%s), falling back to numpy", e)
        return _numpy_col_reductions(arr, col_axis)
