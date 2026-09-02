"""Shared-imports hub for pyutilz.data.polarslib submodules: re-exports common third-party/stdlib names and constants."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

# Pin the logger name to the historical flat-module path so log records keep
# the same channel after the split into a subpackage.
logger = logging.getLogger("pyutilz.data.polarslib")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports -- this module is a shared-imports hub: most names below are unused HERE but re-exported for
# ``columns.py`` / ``aggregations.py`` / ``binning.py`` / ``frames.py`` / ``__init__.py`` to import explicitly from
# ``._common`` (see their ``from ._common import (...)`` blocks), so each import is a deliberate re-export, not dead code.
# ----------------------------------------------------------------------------------------------------------------------------

# Set jemalloc config early
import os

os.environ["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:0"  # prevents memory leak in polars
import polars as pl, polars.selectors as cs  # noqa: F401

from typing import Any, Dict, Iterable, Literal, Optional, Tuple, TypeVar  # noqa: F401
import numpy as np  # noqa: F401


import textwrap  # noqa: F401
from collections import Counter  # noqa: F401
from pyutilz.system.system import clean_ram  # noqa: F401
from pyutilz.core.pythonlib import is_cuda_available, check_cpu_flag  # noqa: F401

# ----------------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------------

POLARS_DEFAULT_NUMAGGS: list = (
    "first last min max mean std arg_max arg_min skew kurtosis entropy n_unique".split()
)  # replace by approx_n_unique? # median excluded
POLARS_DEFAULT_QUANTILES: list = [0.1, 0.25, 0.5, 0.75, 0.9]
