"""``pyutilz.database.db`` -- PostgreSQL/MySQL access layer.

Historically one flat module; carved into cohesive submodules once it passed the 1000-LOC limit.
This ``__init__`` is the FACADE: it owns the mutable module-level connection state and re-exports
every name the flat module exposed, so ``from pyutilz.database.db import X`` and the lazy
``pyutilz.db`` alias keep resolving unchanged (private names such as ``_SQL_IDENTIFIER_RE``,
``_JSON_PATH_RE`` and ``_get_thread_cursors`` included).

Submodules:
  _common      - shared third-party/stdlib imports + the ``pyutilz.database.db`` logger
  connection   - connect_to_db, per-thread cursor cache, cursor accessors, the Insert compiler hook
  execution    - basic_db_execute/safe_execute*, db_command, settings + logging, alchemy raw SQL
  schema       - DDL: table existence/autocreate, id<->name dicts, range partitions, enums, JSON advisor
  regjobs      - the ``regular_jobs`` cooperative scheduler helpers
  legacy       - deprecated PascalCase/Hungarian-notation wrappers
  sql_helpers  - pure identifier validation / quoting / ON CONFLICT clause building
  upsert       - build_upsert_query
  sqlite       - sqlite-flavoured table creation and inserts

The connection state below (``conn``, ``cur``, ``conn_alchemy``, credentials, ``node_id``,
``last_db_settings_read_at``) deliberately lives HERE rather than in a submodule: it is mutated by
``connection.connect_to_db`` and read by every other submodule, so one namespace must own it. The
submodules reach it with ``import pyutilz.database.db as _facade`` plus attribute access AT CALL
TIME -- never ``from pyutilz.database.db import name``, which would both break on the import cycle
and snapshot a stale value.
"""

# ----------------------------------------------------------------------------------------------------------------------------
# Shared module-level names (imports/constants) preserved from the flat module
# ----------------------------------------------------------------------------------------------------------------------------

from ._common import (
    logging,
    logger,
    ensure_installed,
    Any,
    Dict,
    Iterable,
    Optional,
    Union,
    cast,
    re,
    threading,
    warnings,
    json,
    pd,
    sleep,
    Enum,
    join,
    exists,
    weekofmonth,
    datetime_to_utc_timestamp,
    lookup_in_stack,
    sqlalchemy,
    datetime,
    date,
    timezone,
    relativedelta,
    psycopg2,
    sql,
    DuplicateTable,
    ISOLATION_LEVEL_AUTOCOMMIT,
    execute_values,
    Json,
    NamedTupleCursor,
    OperationalError,
    InternalError,
    InterfaceError,
    compiles,
    Insert,
    OnConflictDoNothing,
    PAGE_SIZE,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------------------------------------------------------------
# Annotated (not left to inference from the ``None`` initialiser) because these are rebound at
# runtime -- with a datetime and a SQLAlchemy Engine respectively -- from the submodules.
last_db_settings_read_at: Optional[datetime] = None
conn_alchemy: Optional[Any] = None
# Bound for real by connect_to_db(); declared here so module-level references resolve
# before that first call (and so static analysis can see these names exist at all).
db_flavor: Optional[str] = None
conn: Optional[Any] = None
cur: Optional[Any] = None
db_name: Optional[str] = None
db_host: Optional[str] = None
db_port: Optional[int] = None
db_schema: Optional[str] = None
username: Optional[str] = None
pwd: Optional[str] = None
init_params_fn: Optional[Any] = None
db_sslmode: Optional[str] = None

# Optional override for log_to_db()'s "node" column -- callers may set
# pyutilz.database.db.node_id = "..." directly; falls back to lookup_in_stack()
# when unset (the pre-existing default, unchanged).
node_id: Optional[str] = None

# ----------------------------------------------------------------------------------------------------------------------------
# Re-exported helpers carved into cohesive submodules. The globals above are declared FIRST so that
# a submodule touching them during its own import would still find them bound.
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.database.db.sql_helpers import (
    _SQL_IDENTIFIER_RE,
    validate_sql_identifier,
    validate_sql_qualified_identifier,
    construct_templates_and_values,
    u,
    nu,  # noqa: re-exported facade name, consumed as `pyutilz.database.db.nu`/`pyutilz.db.nu` by external repos, not referenced in-tree
    make_set_excluded_clause,
    MakeSetExcludedClause,
    update_if_now,
)
from pyutilz.database.db.upsert import build_upsert_query  # noqa: re-exported facade name, consumed as `pyutilz.database.db.build_upsert_query` by external repos, not referenced in-tree
from pyutilz.database.db.sqlite import ensure_db_tables_created, insert_sqllite_data  # noqa: ensure_db_tables_created is a re-exported facade name, consumed by external repos, not referenced in-tree

from .connection import (
    prefix_inserts,
    _thread_local,
    _get_thread_cursors,
    connect_to_db,
    get_cursor_type,
    get_cursor,
    mogrify,
)
from .execution import (
    get_table_fields,
    basic_db_execute,
    safe_execute,
    safe_execute_values,
    fetch_db_elements,
    db_command,
    read_db_settings,
    _LOG_TO_DB_KNOWN_LEVELS,
    _JSON_PATH_RE,
    log_to_db,
    explain_table,
    showcase_table,
    select,
    execute_alchemy,
)
from .schema import (
    check_if_pg_table_exists,
    ensure_pg_table_exists,
    read_table_into_dict,
    read_table_into_dict_reversed,
    read_unique_table_field,
    get_id_by_key_field_and_insert_if_needed,
    _iter_partition_dates,
    _partition_table_name,
    create_postgres_range_partitions,
    delete_postgres_range_partitions,
    enable_tables_sizes_approximation,
    create_enum_from_table,
    suggest_json_optimization,
)
from .regjobs import (
    regjobs_create_table,
    regjobs_poll,
    _regjobs_update,
    regjobs_progress,
    regjobs_finalize,
)
from .legacy import (
    EnsurePgTableExists,
    ReadTableIntoDic,
    ReadTableIntoDicReversed,
    GetIdByKeyFieldAndInsertIfNeeded,
)

__all__ = [
    # connection / engine setup
    "connect_to_db",
    "get_cursor",
    "get_cursor_type",
    "mogrify",
    "prefix_inserts",
    "conn",
    "cur",
    "conn_alchemy",
    "db_flavor",
    "db_name",
    "db_host",
    "db_port",
    "db_schema",
    "username",
    "pwd",
    "db_sslmode",
    "init_params_fn",
    "node_id",
    "last_db_settings_read_at",
    "PAGE_SIZE",
    # query execution
    "basic_db_execute",
    "safe_execute",
    "safe_execute_values",
    "get_table_fields",
    "fetch_db_elements",
    "db_command",
    "read_db_settings",
    "log_to_db",
    "select",
    "execute_alchemy",
    "showcase_table",
    "explain_table",
    # DDL / schema
    "check_if_pg_table_exists",
    "ensure_pg_table_exists",
    "read_table_into_dict",
    "read_table_into_dict_reversed",
    "read_unique_table_field",
    "get_id_by_key_field_and_insert_if_needed",
    "create_postgres_range_partitions",
    "delete_postgres_range_partitions",
    "enable_tables_sizes_approximation",
    "create_enum_from_table",
    "suggest_json_optimization",
    # regular jobs
    "regjobs_create_table",
    "regjobs_poll",
    "regjobs_progress",
    "regjobs_finalize",
    # pure SQL helpers / builders
    "validate_sql_identifier",
    "validate_sql_qualified_identifier",
    "construct_templates_and_values",
    "u",
    "nu",
    "make_set_excluded_clause",
    "update_if_now",
    "build_upsert_query",
    "ensure_db_tables_created",
    "insert_sqllite_data",
    # deprecated PascalCase aliases
    "MakeSetExcludedClause",
    "EnsurePgTableExists",
    "ReadTableIntoDic",
    "ReadTableIntoDicReversed",
    "GetIdByKeyFieldAndInsertIfNeeded",
]
