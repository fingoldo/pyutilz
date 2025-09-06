# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

# ensure_installed("sqlalchemy psycopg2 pandas pymysql")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import pandas as pd
import json

from time import sleep
from enum import Enum

from os.path import join,exists

from .pythonlib import weekofmonth, datetime_to_utc_timestamp, lookup_in_stack

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
# Global variables
# ----------------------------------------------------------------------------------------------------------------------------
last_db_settings_read_at = None
connAlchemy = None
cPAGE_SIZE: int = 1_000_000
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
    if type(excluding) == str:
        excluding = excluding.split(",")
    cur.execute("select * from " + table + " where 0=1")
    cur.fetchall()
    if not (cur.description is None):
        return ",".join([alias + "." + col.name + " " + prefix + col.name + suffix for col in cur.description if not (col.name in excluding)])


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
    global db_flavor, conn, cur, cursors, connAlchemy
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
            logger.info("Connecting to the DB %s..." % db_name)

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

                connAlchemy = sqlalchemy.create_engine(conn_string, paramstyle="format")
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

                connAlchemy = sqlalchemy.create_engine(f"mysql+pymysql://{username}:{pwd}@{db_host}:{db_port}/{db_name}")

            if init_params_fn is not None:
                init_params_fn()
        except Exception as e:
            logger.exception(e)
            sleep(5)
        else:
            logger.info("Connected to the DB %s" % db_name)
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
    page_size: int = cPAGE_SIZE,
):
    global cur, cursors

    cursor_type = get_cursor_type(cursor_factory, cursor_name)

    while True:
        try:
            cur = get_cursor(cursor_type=cursor_type, cursor_factory=cursor_factory, cursor_name=cursor_name, itersize=itersize)

            if ex_type == "execute":
                res = cur.execute(statement, data)
            elif ex_type == "execute_values":
                res = execute_values(cur, statement, data, page_size=page_size)

            # if '_named' not in cursor_type:
            # if auto_commit: conn.commit()

        except (OperationalError, InterfaceError) as e:
            logger.exception(e)
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
            if not (cur.description is None):
                # cols_names = [desc[0] for desc in cur.description]
                if return_cursor:
                    return cur
                else:
                    return cur.fetchall()
            else:
                cols_names = []
                if return_cursor:
                    return cur
                else:
                    return []


def safe_execute(statement, data=None, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: int = None):
    return basic_db_execute("execute", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize)


def safe_execute_values(
    statement, data, auto_commit=True, cursor_factory=None, cursor_name=None, return_cursor=False, itersize: int = None, page_size: int = cPAGE_SIZE
):
    return basic_db_execute("execute_values", statement, data, auto_commit, cursor_factory, cursor_name, return_cursor, itersize=itersize, page_size=page_size)


def fetch_db_elements(self, elements, fields, indices=None, prefix=""):
    if type(fields) == str:
        fields = fields.split(",")
    if not (elements is None):
        if fields == ["*"]:
            fields = [col.name for col in cur.description]
        if indices is None:
            indices = range(len(fields))
        for element in elements:
            for i, field in enumerate(fields):
                setattr(self, prefix + field, element[indices[i]])


def construct_templates_and_values(mode, fields, replace_values, source, jsonize):
    """
    Helper sub to assist filling correct templates in db_command sub, based on mode passed.
    """
    values, templates = [], []
    for key in fields:
        if key in replace_values:
            value = replace_values[key]
        else:
            value = source.get(key, None)

        if jsonize:
            if type(value) in (dict, list):
                value = json.dumps(value)

        values.append(value)
        if mode == "insert":
            templates.append("%s")
        elif mode == "select":
            templates.append(key + "=%s")
    return values, templates


def db_command(mode, table_name, where_fields=None, set_fields=None, replace_values={}, returning="*", source={}, jsonize=True, fetch_into=None, prefix=""):
    """
    Executes DML commands easily, looking up sql fields in main and replacement dictionaries, optionally fetching returned values into separate class or dictionary instance[prefixed, if needed]
    This procedure is convenient when you have variables in local or global scope which names match exactly SQL table fields names. Using this sub, in your code you only mention fields names once, pass locals()
    and do not worry about duplicated code! Also it allows fetching returned data into some object in one go. ;-)
    """

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

    if type(where_fields) == str:
        where_fields = where_fields.split(",")
    if type(set_fields) == str:
        set_fields = set_fields.split(",")

    if mode in ["select", "update"]:
        where_values, sql_where_templates = construct_templates_and_values(mode, where_fields, replace_values, source, jsonize)
    if mode in ["insert", "update"]:
        set_values, sql_set_templates = construct_templates_and_values(mode, set_fields, replace_values, source, jsonize)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Carrying out exact sql text
    # ----------------------------------------------------------------------------------------------------------------------------

    if mode == "select":
        sql = "select " + returning + " from  " + table_name + " where " + " and ".join(sql_where_templates)
        sql_fields_values = where_values
    elif mode == "insert":
        sql = "insert into " + table_name + " (" + ",".join(set_fields) + ") values (" + ",".join(sql_set_templates) + ")"
        sql_fields_values = set_values
    elif mode == "update":
        sql = "update  " + table_name + " set " + " and ".join(sql_set_templates) + " where " + " and ".join(sql_where_templates)
        sql_fields_values = set_values + where_values

    if mode in ["insert", "update"]:
        if len(returning) > 0:
            sql += " returning " + returning

    res = safe_execute(sql, sql_fields_values)
    if not (fetch_into is None):
        if prefix == "":
            if table_name.endswith("s"):
                prefix = table_name[:-1] + "_"
        fetch_db_elements(fetch_into, res, returning, prefix=prefix)
    return res


def u(str_val, symb="'"):
    if str_val is None:
        return "null"
    else:
        return symb + str_val.replace(symb, symb * 2) + symb


def nu(str_val, symb="'"):
    if str_val is None or len(str_val) == 0:
        return "null"
    else:
        return u(str_val, symb)


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
        if settings_names_contains:
            sql += " where strpos(name,'%s')>0" % settings_names_contains

        for (setting_name, val, typename) in safe_execute(sql):
            if typename is None:
                typename = "string"
            ltypename = typename.lower()
            if not (val is None):
                if ltypename == "int":
                    val = int(val)
                elif ltypename in ["float", "real", "double", "numeric"]:
                    val = float(val)
                elif ltypename in ["str", "string"]:
                    val = str(val)
                elif ltypename in ["json", "jsonb"]:
                    val = json.loads(str(val))
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


def MakeSetExcludedClause(sFields: str, bAddUpdatedAtTimestamp: Optional[str] = None) -> str:

    res = ""
    v = sFields.split(",")
    for i in v:
        res = res + i + "=excluded." + i + ","
    if bAddUpdatedAtTimestamp:
        res = res + f"{bAddUpdatedAtTimestamp}=(now() at time zone 'utc')"  # updated_at
    else:
        res = res[:-1]
    return res


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
                    "insert into {sTable} ({sAlternateFieldsNames}) values ({sAlternateFieldsValues}) on conflict ({sUniqueConstraintFields}) do update set {MakeSetExcludedClause(sAlternateFieldsNames, bAddUpdatedAtTimestamp)} returning {sIdFieldName}"
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
        return pd.read_sql(f"SHOW FULL COLUMNS FROM {table_name}", con=connAlchemy)["Field Type Comment".split()]


def showcase_table(table_name: str, condition: str = "", limit: int = 5) -> object:
    """Read a sample from a DB table, return as Pandas dataframe"""
    return pd.read_sql(f"SELECT * FROM {table_name}  {condition} limit {limit}", con=connAlchemy)


def select(sql: str) -> object:
    """Execute arbitrary SQL against DB table, return results as Pandas dataframe"""
    return pd.read_sql(sql, con=connAlchemy)


def execute_alchemy(sql: str, max_retries: int = 3) -> object:
    """Execute arbitrary SQL against DB table using Alchemy directly"""
    n = 0
    while n < max_retries:
        try:
            n += 1
            connAlchemy.execute(sql)
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


def suggest_json_optimization(table: str, table_field: str, path: str = "", fields: list = [], min_occurence_percent: float = 0.5, max_vals: int = 5) -> dict:
    """
    Aim is to remove lengthy fields that mostly holds a default value from the JSON.
    Absense of some JSON field is interpreted as null in PostGres.
    Result will be an optimization of an existing table.
    """

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


def build_upsert_query(
    fields_names: list,
    table_name: str,
    conflict_fields: list = ["id"],
    fields_types: dict = {},
    skip_fields: list = [],
    timestamp_check_fields: list = [],
    timestamp_update_fields: list = [],
    on_conflict_update_fields: list = [],
    on_conflict_update_values: dict = {},
    custom_onconflict: str = None,
    returning_fields: list = [],
    history_table_name: str = None,
    history_fields: list = [],
    history_fields_aliases: dict = {},
    hash_fields: str = "",
    default_timestamp: str = "now()",
) -> str:
    """
    Inserts new records or Updates fields in a Postgres table
    Optionally saves data into History table after comparing Hash field of the new and old record.

    Params:
    timestamp_check_fields: when element was last checked (queried from source)
    timestamp_update_fields: when element was last updated (queried from source and found changed)
    """
    # ------------------------------
    # Checks!
    # ------------------------------

    assert len(fields_names) > 0

    for field in timestamp_check_fields:
        assert field not in fields_names
        assert field not in fields_types

    for field_name, field_type in fields_types.items():
        assert field_name in fields_names
        assert field_type in "int bigint smallint float real numeric json jsonb text timestamp".split(" ")

    rev_on_conflict_update_values = {}
    if len(on_conflict_update_values) > 0:
        for field in on_conflict_update_values.keys():
            # assert field in fields_names
            pass
        for field, value in on_conflict_update_values.items():
            rev_on_conflict_update_values[field] = value.replace("excluded.", "")
    # ------------------------------
    # Building query
    # ------------------------------

    query = f"with data({','.join(fields_names)}) as (VALUES %s)"

    if len(timestamp_check_fields) > 0:
        actual_fields_names = fields_names + timestamp_check_fields
    else:
        actual_fields_names = fields_names.copy()

    for field in skip_fields:
        if field in actual_fields_names:
            actual_fields_names.remove(field)

    values = []
    conversion_clause = " at time zone 'utc' " if ("now" in default_timestamp.lower() or "to_timestamp" in default_timestamp.lower()) else ""
    for field in actual_fields_names:
        if field in fields_types:
            values.append(field + "::" + fields_types[field])
        elif field in timestamp_check_fields:
            values.append(f"{default_timestamp}::timestamptz {conversion_clause}")
        else:
            values.append(field)
    fresh_query = f"""
    insert into {table_name} ({','.join(actual_fields_names)}) 
    select {','.join(values)} 
    from data """

    # ------------------------------
    # On conflict
    # ------------------------------

    if custom_onconflict or (conflict_fields is not None and len(conflict_fields) > 0 and len(on_conflict_update_fields) > 0):
        if not custom_onconflict:
            for field in conflict_fields:
                assert field in fields_names

            update_values = []
            for field in on_conflict_update_fields:
                # assert field in actual_fields_names,f"{field} not in actual_fields_names"
                if field in on_conflict_update_values:
                    update_values.append(field + "=" + on_conflict_update_values[field])
                elif field in fields_types:
                    update_values.append(field + "=excluded." + field + "::" + fields_types[field])
                # elif field in timestamp_check_fields:
                #    update_values.append(field + "=" + default_timestamp)
                else:
                    update_values.append(field + "=excluded." + field)

            fresh_query += f"on conflict ({','.join(conflict_fields)}) do update set {','.join(update_values)}"
        else:
            fresh_query += custom_onconflict
    else:
        fresh_query += "on conflict do nothing "

    if len(history_fields) > 0:
        returning_fields = history_fields
        if hash_fields:
            for hash_field in hash_fields:
                assert hash_field in fields_names
                if hash_field not in history_fields:
                    returning_fields += [hash_field]
        fresh_query += f" returning {','.join(returning_fields)}"

    query += ", fresh_data as (" + fresh_query + ")"

    if len(history_table_name) > 0:
        assert len(conflict_fields) > 0
        upd_conflict_fields = "(" + ",".join([f"u.{field}" for field in conflict_fields]) + ")"
        upd_changed_fields = "(" + ",".join([f"c.{field}" for field in conflict_fields]) + ")"

        history_fields_final = [history_fields_aliases.get(field, field) for field in history_fields]
        hist_query = f"""
        ,changed_data as (insert into {history_table_name}({','.join(history_fields_final)}) select {','.join(['u.'+field for field in history_fields])} from fresh_data u
        """
        if len(timestamp_update_fields) > 0:
            the_list = conflict_fields + timestamp_check_fields
        else:
            the_list = conflict_fields

        f_ = [field for field in the_list if field not in history_fields_aliases]
        if len(timestamp_update_fields) > 0:
            for field in timestamp_check_fields:
                new_field = f"{history_fields_aliases.get(field)} as {field}" if field in history_fields_aliases else field
                if new_field not in f_:
                    f_ = f_ + [
                        new_field,
                    ]

        if hash_fields:
            join_condtion = " and ".join([f"u.{field}=c.{field}" for field in conflict_fields])
   

            hash_changing_cond = []
            for hash_field in hash_fields:
                hash_changing_cond.append(
                    f"((c.{hash_field} is null and u.{hash_field} is not null) or (c.{hash_field} is not null and u.{hash_field} is null) or (c.{hash_field}<>u.{hash_field}))"
                )
            hash_changing_cond = " OR ".join(hash_changing_cond)

            hist_query += f"left join {table_name} c on {join_condtion} where ({hash_changing_cond}) returning {','.join(f_)}"
        else:
            join_condtion=None
            hist_query +=f" returning {','.join(f_)}"

        hist_query += ")"
        query += hist_query

        if len(timestamp_update_fields) > 0:
            assert len(timestamp_update_fields) == len(timestamp_check_fields)

            upd_fields_and_vals = ",".join(
                [f"{ufield}=c.{cfield}" for ufield, cfield in zip(timestamp_update_fields, timestamp_check_fields + timestamp_check_fields)]
            )

            # need to figure out name of update field

            if True:
                if join_condtion:
                    the_join_condtion="where {join_condtion}"
                else:
                    the_join_condtion=""

                query += f" select * from changed_data); with tmp as (update {table_name} AS u set {upd_fields_and_vals} from changed_data as c {the_join_condtion}) select count(*) from changed_data;"
            else:
                # query += " select * from changed_data c left join test_agencies t on c.rid=t.rid"
                # query += " update test_agencies as t set info_upd_ts='2021-07-01 00:00:00' from changed_data as c where c.rid=t.rid"
                pass

            query = "create temp table changed_data ON COMMIT DROP as (" + query  # BEGIN TRANSACTION; +" COMMIT;"
        else:
            query += " select 1 from changed_data"
    else:
        query += " select 1 from fresh_data"
    return query


def update_if_now(added_at: str, clause: str) -> str:
    """
    builds an onconflict_clause. Updates existing data only if now is specified in added_at.
    """
    if "now" in added_at.lower():
        onconflict_clause = f"""
            do update set 
                {clause}            
        """
    else:
        onconflict_clause = """do nothing"""

    return onconflict_clause


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

# ----------------------------------------------------------------------------------------------------------------------------
# SQLLITE
# ----------------------------------------------------------------------------------------------------------------------------

def ensure_db_tables_created(cursor:object) -> bool:

    schema_fpath = join("database", "schema.sql")
    if not exists(schema_fpath):
        logger.error(f"DB Schema file not found.")
        return False
    with open(schema_fpath, "r") as f:
        schema_string = f.read()

    if len(schema_string) > 0:
        cursor.executescript(schema_string)
        conn.commit()

        return True
    else:
        logger.error(f"DB Schema empty.")
        return False
    
def insert_sqllite_data(table_name:str,data: Iterable[Dict[str, Any]],columns:Iterable,cursor:object,conn:object,verbose:int=1):
    """Самый быстрый способ для массовых вставок"""
    
    # Создаем SQL запрос
    placeholders = ', '.join(['?' for _ in columns])
    columns_str = ', '.join([f'"{col}"' if col == 'GROUP' else col for col in columns])
    sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
    
    # Преобразуем словари в кортежи в правильном порядке
    values_list = []
    for row in data:
        values = tuple(row.get(col) for col in columns)
        values_list.append(values)
    
    # Вставляем данные
    try:
        cursor.executemany(sql, values_list)
        conn.commit()
        n=len(values_list)
        if verbose: 
            logger.info(f"Inserted {n:_} row(s) into {table_name} table.")
            return n
    except Exception as e:
        logger.error(f"Could not insert data into {table_name} table: {e}.")
        return 0

