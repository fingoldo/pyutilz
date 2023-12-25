"""Nubmba-related utils."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numba
import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


@njit
def set_numba_random_seed(random_seed: int):
    if random_seed is not None:
        np.random.seed(random_seed)


@njit()
def arr2str(arr: Sequence) -> str:
    """Converts a sequence to the textual concatenation of its elements. Analog of str(sequence)."""
    s = ""
    for el in arr:
        s += str(el)
    return s


# ----------------------------------------------------------------------------------------------------------------------------
# Dicts
# ----------------------------------------------------------------------------------------------------------------------------


def python_dict_2_numba_dict(python_dict: dict, numba_dict: numba.typed.Dict()) -> None:
    """THe only way for now is just to copy ke-values: https://github.com/numba/numba/issues/4728"""
    for k, v in python_dict.items():
        numba_dict[k] = v


# ----------------------------------------------------------------------------------------------------------------------------
# Itertools
# ----------------------------------------------------------------------------------------------------------------------------


@njit()
def generate_combinations_recursive_njit(sequence, r):
    """Replacement of itertools combinations function.
    Call generate_combinations_recursive_njit(sequence, r)[::-1] for the same effect as combinations(sequence, r).
    """
    if r == 0:
        return np.empty((1, 0), dtype=sequence.dtype)
    if sequence.size == 0:
        return np.empty((0, r), dtype=sequence.dtype)

    first, rest = sequence[0], sequence[1:]

    without_first = generate_combinations_recursive_njit(rest, r)
    with_first = generate_combinations_recursive_njit(rest, r - 1)

    result = np.empty((without_first.shape[0] + with_first.shape[0], r), dtype=sequence.dtype)

    result[:without_first.shape[0], :] = without_first
    for i in range(with_first.shape[0]):
        result[i + without_first.shape[0], 0] = first
        result[i + without_first.shape[0], 1:] = with_first[i, :]

    return result