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

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports -- this module is a shared-imports hub: most names below are unused HERE but
# re-exported for the sibling submodules and the ``__init__`` facade to import explicitly.
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Iterable, Optional, Union, cast  # noqa: F401
import re  # noqa: F401
import threading  # noqa: F401
import warnings  # noqa: F401
import pandas as pd  # noqa: F401

from time import sleep  # noqa: F401
from enum import Enum  # noqa: F401

from os.path import join, exists  # noqa: F401

from pyutilz.core.pythonlib import weekofmonth, datetime_to_utc_timestamp, lookup_in_stack  # noqa: F401

sqlalchemy: Any
compiles: Any
Insert: Any
OnConflictDoNothing: Any
try:
    import sqlalchemy  # type: ignore[no-redef]
    from sqlalchemy.ext.compiler import compiles  # type: ignore[no-redef]
    from sqlalchemy.sql.expression import Insert  # type: ignore[no-redef]
    from sqlalchemy.dialects.postgresql.dml import OnConflictDoNothing  # type: ignore[no-redef]
except ImportError:  # optional extra 'database': absent in a core-only install
    sqlalchemy = None
    compiles = None
    Insert = None
    OnConflictDoNothing = None
from datetime import datetime, date, timezone  # noqa: F401

relativedelta: Any
try:
    from dateutil.relativedelta import relativedelta  # type: ignore[no-redef]
except ImportError:  # optional extra 'database': absent in a core-only install
    relativedelta = None

psycopg2: Any
sql: Any
DuplicateTable: Any
ISOLATION_LEVEL_AUTOCOMMIT: Any
execute_values: Any
Json: Any
NamedTupleCursor: Any
OperationalError: Any
InternalError: Any
InterfaceError: Any
try:
    import psycopg2  # type: ignore[no-redef]
    from psycopg2 import sql  # type: ignore[no-redef]
    from psycopg2.errors import DuplicateTable  # type: ignore[no-redef]
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # type: ignore[no-redef]
    from psycopg2.extras import execute_values, Json, NamedTupleCursor  # type: ignore[no-redef]
    from psycopg2 import OperationalError, InternalError, InterfaceError  # type: ignore[no-redef]
except ImportError:  # optional extra 'database': absent in a core-only install
    psycopg2 = None
    sql = None
    DuplicateTable = None
    ISOLATION_LEVEL_AUTOCOMMIT = None
    execute_values = None
    Json = None
    NamedTupleCursor = None
    OperationalError = None
    InternalError = None
    InterfaceError = None

# psycopg2.InterfaceError: cursor already closed
# psycopg2.InternalError: current transaction is aborted, commands ignored until end of transaction block

# ----------------------------------------------------------------------------------------------------------------------------
# sqlalchemy tricks
# ----------------------------------------------------------------------------------------------------------------------------

# Default psycopg2 ``execute_values`` page size; a plain constant (never mutated), so submodules may
# bind it directly as a default argument value.
PAGE_SIZE: int = 1_000_000
