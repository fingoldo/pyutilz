# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed

# ensure_installed("sqlalchemy psycopg2 pandas pymysql")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Dict, Iterable, Optional
import pandas as pd
import orjson

from time import sleep
from enum import Enum

from os.path import join, exists

from pyutilz.core.pythonlib import weekofmonth, datetime_to_utc_timestamp, lookup_in_stack

import sqlalchemy
from datetime import datetime, date

from dateutil.relativedelta import relativedelta

import psycopg2
from psycopg2 import sql
from psycopg2.errors import DuplicateTable
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values, NamedTupleCursor, DictCursor
from psycopg2 import OperationalError, InternalError, InterfaceError

# psycopg2.InterfaceError: cursor already closed
# psycopg2.InternalError: current transaction is aborted, commands ignored until end of transaction block
# ----------------------------------------------------------------------------------------------------------------------------
# SQL Injection Protection
# ----------------------------------------------------------------------------------------------------------------------------

import re

# ----------------------------------------------------------------------------------------------------------------------------
# Re-exported PURE / STATELESS helpers carved into cohesive submodules.
# These do NOT read the module-level connection globals (conn/cur/cursors),
# so they live in submodules; the connection-stateful functions below stay
# in this parent module so they all share the SAME global state.
# ``_SQL_IDENTIFIER_RE`` / ``validate_sql_identifier`` are imported here so
# every ``from pyutilz.database.db import X`` and ``pyutilz.db`` alias usage
# keeps resolving exactly as before.
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.database.db.sql_helpers import (
    _SQL_IDENTIFIER_RE,
    validate_sql_identifier,
    construct_templates_and_values,
    u,
    nu,
    MakeSetExcludedClause,
    update_if_now,
)
from pyutilz.database.db.upsert import build_upsert_query
from pyutilz.database.db.sqlite import ensure_db_tables_created, insert_sqllite_data

# ----------------------------------------------------------------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------------------------------------------------------------
last_db_settings_read_at = None
conn_alchemy = None
PAGE_SIZE: int = 1_000_000
# ----------------------------------------------------------------------------------------------------------------------------
# sqlalchemy tricks
# ----------------------------------------------------------------------------------------------------------------------------
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import Insert
from sqlalchemy.dialects.postgresql.dml import OnConflictDoNothing


@compiles(Insert, "postgresql")
def prefix_inserts(insert, compiler, **kw):

    if insert._post_values_clause is None:
        insert._post_values_clause = OnConflictDoNothing()
    return compiler.visit_insert(insert, **kw)


# ----------------------------------------------------------------------------------------------------------------------------


def get_table_fields(table, alias, prefix="", suffix="", excluding=""):
    if isinstance(excluding, str):
        excluding = excluding.split(",")
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table)
    cur.execute("select * from " + table + " where 0=1")
    cur.fetchall()
    if cur.description is not None:
        return ",".join([alias + "." + col.name + " " + prefix + col.name + suffix for col in cur.description if col.name not in excluding])


def connect_to_db(
    m_db_name,
    m_db_host,
    m_db_port,
    m_db_username,
    m_db_pwd,
    m_init_params_fn=None,
    m_db_flavor: str = "postgres",
    m_db_schema: str = None,
    m_db_sslmode: str = None,
):
    global db_flavor, conn, cur, cursors, conn_alchemy
    global db_name, db_host, db_port, db_schema, username, pwd, init_params_fn, db_sslmode
    db_flavor = m_db_flavor
    db_name = m_db_name
    db_host = m_db_host
    db_port = m_db_port
    db_schema = m_db_schema
    username = m_db_username
    pwd = m_db_pwd
    init_params_fn = m_init_params_fn
    db_sslmode = m_db_sslmode

    assert db_flavor in ("postgres", "mysql")

    while True:
        try:
            logger.info("Connecting to the DB %s...", db_name)

            if db_flavor == "postgres":
                #!pip install psycopg2

                params = dict(dbname=db_name, host=db_host, port=db_port, user=username, password=pwd)
                if db_sslmode:
                    params["sslmode"] = db_sslmode
                if db_schema:
                    params["options"] = f"--search_path={db_schema},public"

                conn = psycopg2.connect(**params)
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                conn_string = "postgresql+psycopg2://%s:%s@%s:%s/%s?" % (username, pwd, db_host, db_port, db_name)

                if db_sslmode:
                    conn_string += f"&sslmode={db_sslmode}"
                if db_schema:
                    conn_string += f"&options=--search_path%3D{db_schema},public"

                conn_alchemy = sqlalchemy.create_engine(conn_string, paramstyle="format")
                # https://stackoverflow.com/questions/25917741/sqlalchemy-with-postgres-insert-into-a-table-whose-columns-have-parentheses

                if "cursors" in globals():
                    del cursors
                cursors = dict()

                cur = conn.cursor()

                # if db_schema: cur.execute(f"set search_path to '{db_schema}'")

                cursors["cursor"] = cur
            elif db_flavor == "mysql":
                #!pip install pymysql
                import pymysql

                conn_alchemy = sqlalchemy.create_engine(f"mysql+pymysql://{username}:{pwd}@{db_host}:{db_port}/{db_name}")

            if init_params_fn is not None:
                init_params_fn()
        except Exception as e:
            logger.exception(e)
            sleep(5)
        else:
            logger.info("Connected to the DB %s", db_name)
            return


def get_cursor_type(cursor_factory: object, cursor_name: str = None) -> str:
    if cursor_factory is None:
        cursor_factory = psycopg2.extensions.cursor
    cursor_type = cursor_factory.__name__ + ("" if cursor_name is None else "_named")

    return cursor_type


def get_cursor(cursor_type: str, cursor_factory: object = None, cursor_name: str = None, itersize: int = None) -> object:
    global cursors
    if cursor_type in cursors and "_named" not in cursor_type:
        cur = cursors[cursor_type]
    else:
        cur = conn.cursor(cursor_factory=cursor_factory, name=cursor_name, withhold=(False if cursor_name is None else True))
        if itersize:
            if str(itersize).isnumeric():
                cur.itersize = itersize
        if "_named" not in cursor_type:
            cursors[cursor_type] = cur
    return cur


def mogrify(operation, parameters=None):
    cursor_type = get_cursor_type(None, None)
    cur = get_cursor(cursor_type)
    return cur.mogrify(operation, parameters).decode("utf-8")


def basic_db_execute(
    ex_type,
    statement,
    data=None,
    auto_commit=True,
    cursor_factory=None,
    cursor_name=None,
    return_cursor=False,
    itersize: int = None,
    page_size: int = PAGE_SIZE,
    max_retries: int = 5,
):
    global cur, cursors

    cursor_type = get_cursor_type(cursor_factory, cursor_name)

    # Add circuit breaker to prevent infinite retry loops
    retry_count = 0
    while retry_count < max_retries:
        try:
            cur = get_cursor(cursor_type=cursor_type, cursor_factory=cursor_factory, cursor_name=cursor_name, itersize=itersize)

            if ex_type == "execute":
                cur.execute(statement, data)
            elif ex_type == "execute_values":
                execute_values(cur, statement, data, page_size=page_size)

            # if '_named' not in cursor_type:
            # if auto_commit: conn.commit()

        except (OperationalError, InterfaceError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded for database operation")
                raise
            logger.exception(e)
            logger.info("Retrying database operation (%s/%s)...", retry_count, max_retries)
            sleep(1)
            connect_to_db(
                m_db_name=db_name,
                m_db_host=db_host,
                m_db_port=db_port,
                m_db_username=username,
                m_db_pwd=pwd,
                m_init_params_fn=init_params_fn,
                m_db_flavor=db_flavor,
                m_db_schema=db_schema,
                m_db_sslmode=db_sslmode,
            )
            continue  # Retry the operation
        except DuplicateTable as e:
            logger.warning(e)
            # conn.commit()
            return
        except InternalError as e:
            logger.exception(e)
            # logger.warning("rolling back operation...")
            # conn.rollback()
        except Exception as e:
            logger.exception(e)
            if "cursor" in str(e) and "already exists" in str(e):
                if cursor_type in cursors:
                    del cursors[cursor_type]
                cur.close()
            else:
                raise
        else:
            if cur.description is not None:
                # cols_names = [desc[0] for desc in cur.description]
                if return_cursor:
                    return cur
                else:
                    return cur.fetchall()
            else:
                if return_cursor:
                    return cur
                else:
                    return []


def safe_execute(statement, data=None, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: int = None):
    return basic_db_execute("execute", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize)


def safe_execute_values(
    statement, data, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: int = None, page_size: int = PAGE_SIZE
):
    return basic_db_execute("execute_values", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize, page_size=page_size)


def fetch_db_elements(self, elements, fields, indices=None, prefix=""):
    if isinstance(fields, str):
        fields = fields.split(",")
    if elements is not None:
        if fields == ["*"]:
            fields = [col.name for col in cur.description]
        if indices is None:
            indices = range(len(fields))
        for element in elements:
            for i, field in enumerate(fields):
                setattr(self, prefix + field, element[indices[i]])


def db_command(mode, table_name, where_fields=None, set_fields=None, replace_values=None, returning="*", source=None, jsonize=True, fetch_into=None, prefix=""):
    """
    Executes DML commands easily, looking up sql fields in main and replacement dictionaries, optionally fetching returned values into separate class or dictionary instance[prefixed, if needed]
    This procedure is convenient when you have variables in local or global scope which names match exactly SQL table fields names. Using this sub, in your code you only mention fields names once, pass locals()
    and do not worry about duplicated code! Also it allows fetching returned data into some object in one go. ;-)
    """
    if replace_values is None:
        replace_values = {}
    if source is None:
        source = {}

    # ----------------------------------------------------------------------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------------------------------------------------------------------

    if where_fields is None and set_fields is None:
        logger.error("Either where_fields or set_fields have to be defined")
        return
    if mode not in ["select", "insert", "update"]:
        logger.error("Unknown mode: %s" % mode)
        return

    # ----------------------------------------------------------------------------------------------------------------------------
    # Smart params parsing
    # ----------------------------------------------------------------------------------------------------------------------------

    if isinstance(where_fields, str):
        where_fields = where_fields.split(",")
    if isinstance(set_fields, str):
        set_fields = set_fields.split(",")

    if mode in ["select", "update"]:
        where_values, sql_where_templates = construct_templates_and_values(mode, where_fields, replace_values, source, jsonize)
    if mode in ["insert", "update"]:
        set_values, sql_set_templates = construct_templates_and_values(mode, set_fields, replace_values, source, jsonize)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Carrying out exact sql text
    # ----------------------------------------------------------------------------------------------------------------------------

    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)

    if mode == "select":
        sql = "select " + returning + " from  " + table_name + " where " + " and ".join(sql_where_templates)
        sql_fields_values = where_values
    elif mode == "insert":
        sql = "insert into " + table_name + " (" + ",".join(set_fields) + ") values (" + ",".join(sql_set_templates) + ")"
        sql_fields_values = set_values
    elif mode == "update":
        # Fixed: SET clause must use comma separator, not "and"
        sql = "update  " + table_name + " set " + ", ".join(sql_set_templates) + " where " + " and ".join(sql_where_templates)
        sql_fields_values = set_values + where_values

    if mode in ["insert", "update"]:
        if len(returning) > 0:
            sql += " returning " + returning

    res = safe_execute(sql, sql_fields_values)
    if fetch_into is not None:
        if prefix == "":
            if table_name.endswith("s"):
                prefix = table_name[:-1] + "_"
        fetch_db_elements(fetch_into, res, returning, prefix=prefix)
    return res


def read_db_settings(g, interval_minutes=10, settings_names_contains=None):
    global last_db_settings_read_at
    do_update = False
    if last_db_settings_read_at is None:
        do_update = True
    else:
        now_time = datetime.utcnow()
        if (now_time - last_db_settings_read_at).total_seconds() / 60 >= interval_minutes:
            do_update = True
    if do_update:
        sql = "select name,value,type from settings"
        sql_params = None
        if settings_names_contains:
            # Use parameterized query to prevent SQL injection
            sql += " where strpos(name,%s)>0"
            sql_params = (settings_names_contains,)

        for setting_name, val, typename in safe_execute(sql, sql_params):
            if typename is None:
                typename = "string"
            ltypename = typename.lower()
            if val is not None:
                if ltypename == "int":
                    val = int(val)
                elif ltypename in ["float", "real", "double", "numeric"]:
                    val = float(val)
                elif ltypename in ["str", "string"]:
                    val = str(val)
                elif ltypename in ["json", "jsonb"]:
                    try:
                        val = orjson.loads(val if isinstance(val, (str, bytes, bytearray)) else str(val))
                    except orjson.JSONDecodeError:
                        # Leave non-JSON values untouched rather than crashing settings load.
                        logger.warning("Setting %r has json/jsonb type but value is not valid JSON: %r", setting_name, val)
                elif ltypename in ["bool", "boolean"]:
                    val = val.lower() in ["true", "1", "t", "y", "yes"]
            g[setting_name] = val
        last_db_settings_read_at = datetime.utcnow()


def log_to_db(message, details=None, more_details=None, level="info", append_severity=False, application=None, table_name="logs"):
    import inspect

    cInfo = 1
    cWarning = 2
    cError = 3

    if level:
        s = message
        if details:
            if more_details:
                s = s + " [" + str(details) + "/" + str(more_details) + "]"
            else:
                s = s + " [" + str(details) + "]"
        else:
            if more_details:
                s = s + " [" + str(more_details) + "]"

        if level in ["warning", "warn"]:
            logger.warning(s)
            severity = cWarning
            if append_severity:
                message = "[Warning] " + message
        elif level == "error":
            logger.error(s)
            severity = cError
            if append_severity:
                message = "[ERROR] " + message
        else:
            logger.info(s)
            severity = cInfo

        node = globals().get("node_id")
        if node is None or application is None:
            if node is None:
                node = lookup_in_stack("node_id")
            if application is None:
                application = lookup_in_stack("app_name")

        safe_execute(
            "insert into " + table_name + "(node,message,details,more_details,severity,application) values (%s,%s,%s,%s,%s,%s)",
            (node, message, details, more_details, severity, application),
        )


def check_if_pg_table_exists(table_name: str, schema_name: Optional[str] = "public"):
    res = safe_execute(
        f"""
    SELECT EXISTS (
       SELECT FROM information_schema.tables
       WHERE  table_schema = {u(schema_name)} AND table_name={u(table_name)}
   )"""
    )
    if res:
        return res[0][0]


def EnsurePgTableExists(sTable: str, sKeyFieldName: Optional[str] = "name", sIdFieldName: Optional[str] = "id", sAutocreateIdTypeName: Optional[str] = None):
    if not check_if_pg_table_exists(sTable):
        if sAutocreateIdTypeName:
            assert sAutocreateIdTypeName.lower() in ("smallserial serial bigserial uuid".split())
            default_gen = " default gen_random_uuid()" if sAutocreateIdTypeName == "uuid" else ""
            safe_execute(
                f"create table {sTable} ({sIdFieldName} {sAutocreateIdTypeName} primary key {default_gen},{sKeyFieldName} text, added_at timestamp without time zone DEFAULT (now() at time zone 'utc'))"
            )


def ReadTableIntoDic(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """
    Reads id->value mapping into a dictionary
    if sAutocreateIdTypeName is specified, if table does not exist, it gets created with specified key type
    """

    dicEnums.clear()
    EnsurePgTableExists(sTable=sTable, sKeyFieldName=sKeyFieldName, sIdFieldName=sIdFieldName, sAutocreateIdTypeName=sAutocreateIdTypeName)
    res = safe_execute(f"select {sIdFieldName},{sKeyFieldName} from {sTable} {sCondition}")
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dicEnums[key] = the_id


def ReadTableIntoDicReversed(
    dicEnums: dict,
    sTable: str,
    sKeyFieldName: Optional[str] = "name",
    sCondition: Optional[str] = "",
    sIdFieldName: Optional[str] = "id",
    sAutocreateIdTypeName: Optional[str] = None,
) -> None:
    """Reads value->id mapping into a dictionary"""

    dicEnums.clear()
    EnsurePgTableExists(sTable=sTable, sKeyFieldName=sKeyFieldName, sIdFieldName=sIdFieldName, sAutocreateIdTypeName=sAutocreateIdTypeName)
    res = safe_execute(f"select {sIdFieldName},{sKeyFieldName} from {sTable} {sCondition}")
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dicEnums[the_id] = key


def read_unique_table_field(table_name: str, field_name: str, container: object, clear: bool = True, placeholder_value=None) -> object:
    if clear:
        container.clear()
    res = safe_execute(
        sql.SQL("select distinct {field_name} from {table_name}").format(field_name=sql.Identifier(field_name), table_name=sql.Identifier(table_name))
    )
    if isinstance(container, set):
        for rec in res:
            container.add(rec[0])
    elif isinstance(container, dict):
        for rec in res:
            container[rec[0]] = placeholder_value
    return container


def GetIdByKeyFieldAndInsertIfNeeded(
    dicEnums: dict,
    sTable: str,
    sKeyFieldValue: str,
    sKeyFieldName: Optional[str] = "name",
    bKeyIsNotString: Optional[bool] = False,
    sAlternateFieldsNames: Optional[str] = "",
    sAlternateFieldsValues: Optional[str] = "",
    sUniqueConstraintFields: Optional[str] = "",
    bUseAlternateFieldsOnly: Optional[bool] = False,
    sIdFieldName: Optional[str] = "id",
    bAddUpdatedAtTimestamp: Optional[str] = None,
) -> str:

    if sKeyFieldValue == "null":
        return "null"

    if sKeyFieldValue in dicEnums:
        return dicEnums[sKeyFieldValue]
    else:
        if sUniqueConstraintFields == "":
            sUniqueConstraintFields = sKeyFieldName

        if bKeyIsNotString:
            Data = sKeyFieldValue
        else:
            Data = u(sKeyFieldValue)
        if len(sAlternateFieldsNames) > 0:
            if not bUseAlternateFieldsOnly:
                rs = safe_execute(
                    f"insert into {sTable} ({sKeyFieldName} , {sAlternateFieldsNames}) values ({Data},{sAlternateFieldsValues}) on conflict ({sUniqueConstraintFields}) do update set {MakeSetExcludedClause(sKeyFieldName, bAddUpdatedAtTimestamp)} returning {sIdFieldName}"
                )
            else:
                rs = safe_execute(
                    f"insert into {sTable} ({sAlternateFieldsNames}) values ({sAlternateFieldsValues}) on conflict ({sUniqueConstraintFields}) do update set {MakeSetExcludedClause(sAlternateFieldsNames, bAddUpdatedAtTimestamp)} returning {sIdFieldName}"
                )
        else:
            rs = safe_execute(
                f"insert into {sTable} ({sKeyFieldName}) values ({Data}) on conflict ({sUniqueConstraintFields}) do update set {MakeSetExcludedClause(sKeyFieldName, bAddUpdatedAtTimestamp)} returning {sIdFieldName}"
            )

        the_id = rs[0][0]
        dicEnums[sKeyFieldValue] = the_id

        return the_id


def create_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str, bigint_degree: int = 0):

    assert partition_size in ("day", "week", "month", "year")
    d = from_date
    while d <= to_date:
        if partition_size == "day":
            n = d + relativedelta(days=1)
        elif partition_size == "week":
            n = d + relativedelta(weeks=1)
        elif partition_size == "month":
            n = d + relativedelta(months=1)
        elif partition_size == "year":
            n = d + relativedelta(years=1)
        if bigint_degree is None or bigint_degree == 0:
            cmd = f"CREATE TABLE z_{table_name}_y{d.year:0000d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d} PARTITION OF {table_name} FOR VALUES FROM ('{d:%Y-%m-%d %H:%M:%S}') TO ('{n:%Y-%m-%d %H:%M:%S}')"
        else:
            cmd = f"CREATE TABLE z_{table_name}_y{d.year:0000d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d} PARTITION OF {table_name} FOR VALUES FROM ('{datetime_to_utc_timestamp(d)*int(10**bigint_degree)}') TO ('{datetime_to_utc_timestamp(n)*int(10**bigint_degree)}')"
        # print(cmd)
        safe_execute(cmd)
        d = n


def delete_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str) -> None:

    assert partition_size in ("day", "week", "month", "year")
    d = from_date
    while d <= to_date:
        if partition_size == "day":
            n = d + relativedelta(days=1)
        elif partition_size == "week":
            n = d + relativedelta(weeks=1)
        elif partition_size == "month":
            n = d + relativedelta(months=1)
        elif partition_size == "year":
            n = d + relativedelta(years=1)

        cmd = f"drop table z_{table_name}_y{d.year:0000d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d}"
        safe_execute(cmd)
        d = n


def explain_table(table_name: str) -> object:
    """Read table names along with comments from a DB table, return as Pandas dataframe"""
    if db_flavor == "mysql":
        return pd.read_sql(f"SHOW FULL COLUMNS FROM {table_name}", con=conn_alchemy)["Field Type Comment".split()]


def showcase_table(table_name: str, condition: str = "", limit: int = 5) -> object:
    """Read a sample from a DB table, return as Pandas dataframe.

    WARNING: The 'condition' parameter should be a safe WHERE clause expression.
    For user input, use parameterized queries instead of passing raw SQL conditions.
    """
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    # Note: condition parameter should ideally be deprecated in favor of parameterized queries
    # but kept for backward compatibility with warning
    if condition and not condition.strip().lower().startswith('where'):
        condition = 'WHERE ' + condition
    return pd.read_sql(f"SELECT * FROM {table_name} {condition} LIMIT {int(limit)}", con=conn_alchemy)


def select(sql: str) -> object:
    """Execute arbitrary SQL against DB table, return results as Pandas dataframe"""
    return pd.read_sql(sql, con=conn_alchemy)


def execute_alchemy(sql: str, max_retries: int = 3) -> object:
    """Execute arbitrary SQL against DB table using Alchemy directly"""
    n = 0
    while n < max_retries:
        try:
            n += 1
            conn_alchemy.execute(sql)
            break
        except Exception as e:
            logger.exception(e)
            sleep(3)


def enable_tables_sizes_approximation():
    """
    Example usage after the funciton is crated:

    select table_name,nrows
        FROM
            cozmee.get_approximate_tables_sizes(
                array['public.amazon_products','public.amazon_users_reviews','public.amazon_products_reviews']
                )

    """

    safe_execute(
        """

CREATE or replace FUNCTION get_approximate_tables_sizes(tables_names text[])
RETURNS  TABLE (table_name text,nrows bigint)  AS
$func$

	SELECT relname::text,reltuples::bigint AS estimate FROM pg_class WHERE  oid in (select unnest($1)::regclass)
$func$ LANGUAGE sql;
"""
    )


def create_enum_from_table(enum_name: str, table_name: str, id_field_name: str, desc_field_name: str) -> object:
    """Uses values and ids from a PG database to fill in Python enum.

    LinksTypes=create_enum_from_table("LinksTypes",table_name="links_types",id_field_name="id",desc_field_name="name")
    """
    dct = {}
    for id_field, desc_field in safe_execute(
        sql.SQL("select {id_field},{desc_field} from {table}").format(
            table=sql.Identifier(table_name), id_field=sql.Identifier(id_field_name), desc_field=sql.Identifier(desc_field_name)
        )
    ):
        dct[desc_field.upper()] = id_field
    return Enum(enum_name, dct)


def suggest_json_optimization(table: str, table_field: str, path: str = "", fields: list = None, min_occurence_percent: float = 0.5, max_vals: int = 5) -> dict:
    """
    Aim is to remove lengthy fields that mostly holds a default value from the JSON.
    Absense of some JSON field is interpreted as null in PostGres.
    Result will be an optimization of an existing table.
    """
    if fields is None:
        fields = []

    # Кандидат должен быть лидером по встречаемости. Также должен покрывать как минимум 90% вариантов. Кандидат не должн быть не-None, если поле имеет None в уже существующих значениях:
    # minHoursWeek [(None, 1244198), ('0', 150675)] - OK
    # risingTalent [(None, 97558), ('true', 441321), ('false', 856003)] - FAIL. false мог бы быть хорошим кандидатом, но уже встречается None.

    res = {}

    for field in fields:

        # Prepare params
        if path == "" or path is None:
            full_path = table_field
        else:
            full_path = table_field + "->" + path

        # Ask DB
        vals = safe_execute(
            f"""
                    select {full_path}->>'{field}' as val,count(*) as qty from {table} where {full_path} is not null group by {full_path}->>'{field}' order by qty desc
            """
        )

        # Compute distributiuon %
        has_none = False
        max_occurences = -1
        total_occurences = 0

        for val, occs in vals:
            if val is None:
                has_none = True
            total_occurences += occs
            if occs > max_occurences:
                max_occurences = occs
                best_val = val

        # Print native values, if not too many
        if len(vals) <= max_vals:
            print(field, vals)
        else:
            print(field, vals[:max_vals], "...")

        # Print opt suggestions for that field
        if total_occurences > 0 and max_occurences / total_occurences >= min_occurence_percent:
            if has_none and (best_val is not None):
                print(" ")
            else:
                print(f"Suggestion: make {best_val} as default for {field}. That will save {max_occurences/total_occurences:.2%} of space.")

                res[field] = best_val
        else:
            print(" ")

    return res


def regjobs_create_table(table_name: str = "regular_jobs"):
    safe_execute(
        sql.SQL(
            """
create table {table_name} (
	name citext primary key,
	description text,
	interval_minutes int,
	singleton boolean default true, --do not start new instance if prev one is still running (unless ping has timed out)
	ping_timeout_minutes int,

	taken_by text, --node ip

	started_at timestamp without time zone,

	last_ping_at timestamp without time zone,
	last_result jsonb,

	finished_at timestamp without time zone,
	result jsonb

	)
    """
        ).format(table_name=sql.Identifier(table_name))
    )


def regjobs_poll(job_name: str, taken_by: str, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL(
            """
        with base as (select name from {table_name} where name=%(job_name)s
            and
                (
                    (started_at is null) --never started before
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at>started_at)
                            and
                        singleton=true
                    ) -- not finished long enough after successful finishing for a singleton
                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        (finished_at is null or finished_at<started_at)
                            and
                        (ping_timeout_minutes is not null and (last_ping_at is null or last_ping_at<(now() at time zone 'utc'-ping_timeout_minutes * interval '1 minute')))
                            and
                        singleton=true
                    ) -- not finished long enough WITHOUT successful finishing yet AND with a missed ping, for a singleton


                        or
                    (
                        (finished_at is null or finished_at <(now() at time zone 'utc'-interval_minutes * interval '1 minute'))
                            and
                        singleton=false
                    ) -- not finished long enough for NOT a singleton

                )
                limit 1 for update skip locked)

            update {table_name} rj set taken_by=%(taken_by)s,started_at=now() at time zone 'utc' from base where rj.name=base.name returning rj.name
    """
        ).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "taken_by": taken_by},
    )


def regjobs_progress(job_name: str, result: dict, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL(
            """
        update {table_name} set last_ping_at=now() at time zone 'utc',last_result=%(result)s where name=%(job_name)s
    """
        ).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "result": result},
    )


def regjobs_finalize(job_name: str, result: dict, table_name: str = "regular_jobs"):
    return safe_execute(
        sql.SQL(
            """
        update {table_name} set finished_at=now() at time zone 'utc',result=%(result)s where name=%(job_name)s
    """
        ).format(table_name=sql.Identifier(table_name)),
        {"job_name": job_name, "result": result},
    )
