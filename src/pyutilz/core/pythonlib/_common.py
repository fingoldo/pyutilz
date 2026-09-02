"""Shared-imports hub for the ``pyutilz.core.pythonlib`` submodules.

Holds the logger plus every stdlib/third-party name the flat module bound at top level, so each
submodule (and the package facade) imports them from one place and the facade's attribute surface
stays byte-for-byte what the flat module exposed.
"""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger("pyutilz.core.pythonlib")  # pinned to the historical flat-module path so log records keep the same channel after the split


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Set, Tuple, Union
from collections.abc import Mapping as MappingABC, Sized as SizedABC

import time
import numbers
import inspect
from numba import njit
from datetime import datetime, date, timezone

from os.path import abspath, exists, join
from contextlib import contextmanager
from functools import lru_cache
import os, sys
import joblib
import errno

import re

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

import importlib.util, subprocess  # nosec B404 - only used below to shell out to the fixed "pip" executable for installing missing packages; no user/network input reaches argv

# ----------------------------------------------------------------------------------------------------------------------------
# Shelve
# ----------------------------------------------------------------------------------------------------------------------------


import shelve  # nosec B403 - open_safe_shelve below only opens a locally-supplied, lock-protected db_path; no untrusted/network data is ever unpickled from an unknown source
import contextlib
import portalocker

# Declared explicitly because most of these names are unused HERE by construction -- this module
# exists so the split submodules and the package facade bind them from one place, reproducing the
# flat module's attribute surface exactly. Without __all__ they read as dead imports to the linter.
__all__ = [
    "logger",
    "Any",
    "Callable",
    "Iterable",
    "Literal",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Union",
    "MappingABC",
    "SizedABC",
    "logging",
    "time",
    "numbers",
    "inspect",
    "njit",
    "datetime",
    "date",
    "timezone",
    "abspath",
    "exists",
    "join",
    "contextmanager",
    "lru_cache",
    "os",
    "sys",
    "joblib",
    "errno",
    "re",
    "importlib",
    "subprocess",
    "shelve",
    "contextlib",
    "portalocker",
]
