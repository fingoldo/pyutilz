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

from pyutilz.core.pythonlib import ensure_installed

# ensure_installed("pandas numpy pyarrow")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Union, Optional, Sequence, Dict, List, Tuple

import gc
import io
import os
import warnings
import numpy as np
import pandas as pd, polars as pl
from pyutilz.text.strings import find_between
from collections import defaultdict
from pyarrow.dataset import dataset
from pyutilz.pythonlib import to_float

import ctypes
from multiprocessing import Array

# Make IPython optional
try:
    from IPython.display import display, Markdown, Latex
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from pyutilz.system.system import tqdmu
from os.path import join, sep
import glob

from os.path import basename, dirname, splitext, exists, getsize
from pyutilz.system.system import ensure_dir_exists
from timeit import default_timer as timer
from itertools import chain
import tempfile
import shutil
