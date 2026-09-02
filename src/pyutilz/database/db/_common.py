"""Shared-imports hub for the ``pyutilz.database.db`` subpackage.

Every third-party/stdlib name the pre-split flat module bound at module level lives here, so the
submodules (``connection``/``execution``/``schema``/``regjobs``/``legacy``) and the ``__init__``
facade all import them from ONE place and the facade's attribute surface stays identical.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

# Pinned to the historical flat-module channel so log records (and caplog filters) keep working
# unchanged after the carve into submodules.
logger = logging.getLogger("pyutilz.database.db")

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed  # noqa: F401

# ensure_installed("sqlalchemy psycopg2 pandas pymysql")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports -- this module is a shared-imports hub: most names below are unused HERE but
# re-exported for the sibling submodules and the ``__init__`` facade to import explicitly.
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Iterable, Optional, Union, cast  # noqa: F401
import re  # noqa: F401
import threading  # noqa: F401
import warnings  # noqa: F401
import json  # noqa: F401
import pandas as pd  # noqa: F401

from time import sleep  # noqa: F401
from enum import Enum  # noqa: F401

from os.path import join, exists  # noqa: F401

from pyutilz.core.pythonlib import weekofmonth, datetime_to_utc_timestamp, lookup_in_stack  # noqa: F401

import sqlalchemy  # noqa: F401
from datetime import datetime, date, timezone  # noqa: F401

from dateutil.relativedelta import relativedelta  # noqa: F401

import psycopg2  # noqa: F401
from psycopg2 import sql  # noqa: F401
from psycopg2.errors import DuplicateTable  # noqa: F401
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # noqa: F401
from psycopg2.extras import execute_values, Json, NamedTupleCursor  # noqa: F401
from psycopg2 import OperationalError, InternalError, InterfaceError  # noqa: F401

# psycopg2.InterfaceError: cursor already closed
# psycopg2.InternalError: current transaction is aborted, commands ignored until end of transaction block

# ----------------------------------------------------------------------------------------------------------------------------
# sqlalchemy tricks
# ----------------------------------------------------------------------------------------------------------------------------
from sqlalchemy.ext.compiler import compiles  # noqa: F401
from sqlalchemy.sql.expression import Insert  # noqa: F401
from sqlalchemy.dialects.postgresql.dml import OnConflictDoNothing  # noqa: F401

# Default psycopg2 ``execute_values`` page size; a plain constant (never mutated), so submodules may
# bind it directly as a default argument value.
PAGE_SIZE: int = 1_000_000
