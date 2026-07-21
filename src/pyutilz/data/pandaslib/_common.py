"""Shared-imports hub for pyutilz.data.pandaslib submodules: re-exports common third-party/stdlib names."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

# Pin the logger name to the historical flat-module path so log records keep
# the same channel after the split into a subpackage.
logger = logging.getLogger("pyutilz.data.pandaslib")

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed  # noqa: F401

# ensure_installed("pandas numpy pyarrow")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports -- this module is a shared-imports hub: every name below is unused HERE but re-exported for
# ``benchmarks.py`` / ``dtypes.py`` / ``frames.py`` / ``io_ops.py`` / ``__init__.py`` to import explicitly from
# ``._common`` (see their ``from ._common import (...)`` blocks), so each import is a deliberate re-export, not dead code.
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Union, Optional, Sequence, Dict, List, Tuple  # noqa: F401

import gc  # noqa: F401
import io  # noqa: F401
import os  # noqa: F401
import warnings  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd, polars as pl  # noqa: F401
from pyutilz.text.strings import find_between  # noqa: F401
from collections import defaultdict  # noqa: F401
from pyarrow.dataset import dataset  # noqa: F401
from pyutilz.core.pythonlib import to_float  # noqa: F401

import ctypes  # noqa: F401
from multiprocessing import Array  # noqa: F401

# Make IPython optional
try:
    from IPython.display import display, Markdown, Latex  # noqa: F401
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from pyutilz.system.system import tqdmu  # noqa: F401
from os.path import join, sep  # noqa: F401
import glob  # noqa: F401

from os.path import basename, dirname, splitext, exists, getsize  # noqa: F401
from pyutilz.system.system import ensure_dir_exists  # noqa: F401
from timeit import default_timer as timer  # noqa: F401
from itertools import chain  # noqa: F401
import tempfile  # noqa: F401
import shutil  # noqa: F401
