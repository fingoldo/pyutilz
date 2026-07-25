"""Nubmba-related utils."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Sequence

import numba
import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


@njit
def set_numba_random_seed(random_seed: int):
    """Set random seed for numpy random number generator in numba context.

    Note: This function requires a seed value. For convenience with None defaults,
    use the wrapper function set_random_seed() instead.
    """
    np.random.seed(random_seed)  # noqa: NPY002 -- this is a @njit kernel; np.random.Generator is not nopython-mode compatible, same as mlframe's identical numba-interop exception


def set_random_seed(random_seed: Optional[int] = None):
    """Convenience wrapper that handles None defaults before calling numba function."""
    if random_seed is None:
        random_seed = 42
    set_numba_random_seed(random_seed)


def arr2str(arr: Sequence) -> str:
    """Converts a sequence to the textual concatenation of its elements. Analog of str(sequence).

    Note: This function is NOT compiled with @njit because string operations in numba
    are slow and use O(N²) concatenation. For performance-critical code, avoid string
    operations in numba contexts.
    """
    return "".join(str(el) for el in arr)


# ----------------------------------------------------------------------------------------------------------------------------
# Dicts
# ----------------------------------------------------------------------------------------------------------------------------


def python_dict_2_numba_dict(python_dict: dict, numba_dict: Optional[numba.typed.Dict] = None) -> numba.typed.Dict:
    """THe only way for now is just to copy key-values: https://github.com/numba/numba/issues/4728"""
    if numba_dict is None:
        numba_dict = numba.typed.Dict()
    for k, v in python_dict.items():
        numba_dict[k] = v
    return numba_dict


# ----------------------------------------------------------------------------------------------------------------------------
# Itertools
# ----------------------------------------------------------------------------------------------------------------------------


@njit()
def _generate_combinations_recursive_njit_core(sequence, r):
    """Recursive core of generate_combinations_recursive_njit; assumes r >= 0 (validated by the caller)."""
    if r == 0:
        return np.empty((1, 0), dtype=sequence.dtype)
    if sequence.size == 0:
        return np.empty((0, r), dtype=sequence.dtype)

    first, rest = sequence[0], sequence[1:]

    without_first = _generate_combinations_recursive_njit_core(rest, r)
    with_first = _generate_combinations_recursive_njit_core(rest, r - 1)

    result = np.empty((without_first.shape[0] + with_first.shape[0], r), dtype=sequence.dtype)

    result[: without_first.shape[0], :] = without_first
    for i in range(with_first.shape[0]):
        result[i + without_first.shape[0], 0] = first
        result[i + without_first.shape[0], 1:] = with_first[i, :]

    return result


@njit(cache=True)
def generate_combinations_recursive_njit(sequence, r):
    """Replacement of itertools combinations function.
    Call generate_combinations_recursive_njit(sequence, r)[::-1] for the same effect as combinations(sequence, r).

    ``@njit`` (restored): a prior revision made this a plain-Python wrapper around the njit-decorated
    ``_generate_combinations_recursive_njit_core`` to add the ``r < 0`` validation. That silently broke every
    external caller that invokes this function from ITS OWN ``@njit`` context (e.g. mlframe's
    ``evaluate_gain``) -- numba's nopython mode cannot call a plain Python function, so those callers failed
    with ``TypingError: Untyped global name ... Cannot determine Numba type of <class 'function'>``. Restored
    as an njit function with the same validation (numba nopython mode supports raising with a static string
    message); still callable from plain Python exactly as before.
    """
    if r < 0:
        raise ValueError("generate_combinations_recursive_njit: r must be >= 0")
    return _generate_combinations_recursive_njit_core(sequence, r)
