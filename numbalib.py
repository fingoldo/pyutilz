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
def set_random_seed(random_seed: int):
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
