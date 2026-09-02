"""DDL / schema helpers for ``pyutilz.database.db``.

Table existence + autocreation, id<->name lookup dictionaries, range-partition create/drop, enum
generation from a table, and the JSON-column optimization advisor. Everything here reaches the DB
through the facade's ``safe_execute``, so it carries no connection state of its own.
"""

from typing import Optional, Union

from ._common import (
    logger,
    date,
    sql,
    weekofmonth,
    datetime_to_utc_timestamp,
    relativedelta,
    Enum,
)
from .sql_helpers import make_set_excluded_clause, u, validate_sql_identifier

# See the PROJECT IDIOM comment in connection.py.
import pyutilz.database.db as _facade


def check_if_pg_table_exists(table_name: str, schema_name: Optional[str] = "public"):
    """True when ``table_name`` exists in ``schema_name`` according to ``information_schema.tables``."""
    # table_name/schema_name are compared against information_schema metadata columns as VALUES,
    # not identifiers -- parameterized via %s placeholders (correctness/consistency fix: this
    # previously used u()'s manual quote-doubling escape instead of driver-level parameter
    # binding, unlike log_to_db()'s equivalent query a few lines above in the same file, which
    # already used %s correctly for the same class of comparison).
    res = _facade.safe_execute(
        """
    SELECT EXISTS (
       SELECT FROM information_schema.tables
       WHERE  table_schema = %s AND table_name = %s
   )""",
        (schema_name, table_name),
    )
    if res:
        return res[0][0]


def ensure_pg_table_exists(
    table: str, key_field_name: Optional[str] = "name", id_field_name: Optional[str] = "id", autocreate_id_type_name: Optional[str] = None
) -> None:
    """Create ``table`` (with ``key_field_name``/``id_field_name`` columns) if it doesn't exist yet.

    Also importable as :func:`EnsurePgTableExists` -- a deprecated alias, same function, kept for
    backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """
    # Validate identifiers to prevent SQL injection
    validate_sql_identifier(table)
    validate_sql_identifier(key_field_name)
    validate_sql_identifier(id_field_name)
    if not _facade.check_if_pg_table_exists(table):
        if autocreate_id_type_name:
            if autocreate_id_type_name.lower() not in ("smallserial serial bigserial uuid".split()):
                # autocreate_id_type_name is spliced verbatim into the CREATE TABLE statement below with no other
                # validation, so under `python -O` a skipped assert would let arbitrary SQL be injected via this arg.
                raise ValueError(f"Invalid autocreate_id_type_name: {autocreate_id_type_name!r}")
            # Regression fix: the validation above (2 lines up) correctly lowercases before
            # comparing against the whitelist, but this comparison didn't -- autocreate_id_type_name="UUID"
            # passed validation ("UUID".lower() == "uuid" is in the whitelist) but silently failed
            # THIS check ("UUID" == "uuid" is False), so `default_gen` stayed empty and the
            # generated DDL had no `default gen_random_uuid()` despite auto-generation having
            # been explicitly requested -- any insert omitting the id column (the whole point of
            # autocreate_id_type_name) would then fail with a NOT NULL constraint violation.
            default_gen = " default gen_random_uuid()" if autocreate_id_type_name.lower() == "uuid" else ""
            _facade.safe_execute(
                f"create table {table} ({id_field_name} {autocreate_id_type_name} primary key {default_gen},{key_field_name} text, added_at timestamp without time zone DEFAULT (now() at time zone 'utc'))"
            )


def read_table_into_dict(
    dict_enums: dict,
    table: str,
    key_field_name: Optional[str] = "name",
    condition: Optional[str] = "",
    id_field_name: Optional[str] = "id",
    autocreate_id_type_name: Optional[str] = None,
) -> None:
    """
    Reads id->value mapping into a dictionary
    if autocreate_id_type_name is specified, if table does not exist, it gets created with specified key type

    Also importable as :func:`ReadTableIntoDic` -- a deprecated alias, same function, kept for
    backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    dict_enums.clear()
    _facade.ensure_pg_table_exists(table=table, key_field_name=key_field_name, id_field_name=id_field_name, autocreate_id_type_name=autocreate_id_type_name)
    # table/key_field_name/id_field_name validated inside ensure_pg_table_exists above; condition is an accepted raw WHERE fragment
    res = _facade.safe_execute(f"select {id_field_name},{key_field_name} from {table} {condition}")  # nosec B608
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dict_enums[key] = the_id


def read_table_into_dict_reversed(
    dict_enums: dict,
    table: str,
    key_field_name: Optional[str] = "name",
    condition: Optional[str] = "",
    id_field_name: Optional[str] = "id",
    autocreate_id_type_name: Optional[str] = None,
) -> None:
    """Reads value->id mapping into a dictionary.

    WARNING: condition is spliced verbatim into the SQL statement (raw WHERE fragment,
    unvalidated). This function executes raw, unvalidated SQL - condition must NEVER
    be built from external/user-controlled input directly; only pass trusted, hard-coded
    or internally-constructed condition strings.

    Also importable as :func:`ReadTableIntoDicReversed` -- a deprecated alias, same function,
    kept for backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    dict_enums.clear()
    _facade.ensure_pg_table_exists(table=table, key_field_name=key_field_name, id_field_name=id_field_name, autocreate_id_type_name=autocreate_id_type_name)
    # table/key_field_name/id_field_name validated inside ensure_pg_table_exists above; condition is an accepted raw WHERE fragment
    res = _facade.safe_execute(f"select {id_field_name},{key_field_name} from {table} {condition}")  # nosec B608
    for rs in res:
        if rs[1] is not None:
            if rs[0] is not None:
                the_id = rs[0]
                key = rs[1]
                dict_enums[the_id] = key


def read_unique_table_field(table_name: str, field_name: str, container: Union[set, dict], clear: bool = True, placeholder_value=None) -> Union[set, dict]:
    """Loads the distinct values of one table column into ``container`` and returns it.

    A set receives the values themselves; a dict receives them as keys, each mapped to
    ``placeholder_value``. ``clear`` empties the container first, so repeated calls refresh rather
    than accumulate.
    """
    if clear:
        container.clear()
    res = _facade.safe_execute(
        sql.SQL("select distinct {field_name} from {table_name}").format(field_name=sql.Identifier(field_name), table_name=sql.Identifier(table_name))
    )
    if isinstance(container, set):
        for rec in res:
            container.add(rec[0])
    elif isinstance(container, dict):
        for rec in res:
            container[rec[0]] = placeholder_value
    return container


def get_id_by_key_field_and_insert_if_needed(
    dict_enums: dict,
    table: str,
    key_field_value: str,
    key_field_name: Optional[str] = "name",
    key_is_not_string: Optional[bool] = False,
    alternate_fields_names: Optional[str] = "",
    alternate_fields_values: Optional[str] = "",
    unique_constraint_fields: Optional[str] = "",
    use_alternate_fields_only: Optional[bool] = False,
    id_field_name: Optional[str] = "id",
    add_updated_at_timestamp: Optional[str] = None,
) -> str:
    """Look up ``key_field_value``'s id in ``dict_enums`` (or the DB), inserting a new row if needed.

    Also importable as :func:`GetIdByKeyFieldAndInsertIfNeeded` -- a deprecated alias, same
    function, kept for backward compatibility with the legacy PascalCase/Hungarian-notation name.
    """

    if key_field_value == "null":
        return "null"

    if key_field_value in dict_enums:
        return dict_enums[key_field_value]  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    else:
        key_field_name = key_field_name if key_field_name is not None else "name"
        alternate_fields_names = alternate_fields_names if alternate_fields_names is not None else ""
        unique_constraint_fields = unique_constraint_fields if unique_constraint_fields is not None else ""
        id_field_name = id_field_name if id_field_name is not None else "id"

        # Validate identifiers to prevent SQL injection
        validate_sql_identifier(table)
        validate_sql_identifier(key_field_name)
        validate_sql_identifier(id_field_name)
        for _name in [n for n in alternate_fields_names.split(",") if n]:
            validate_sql_identifier(_name)
        if unique_constraint_fields == "":
            unique_constraint_fields = key_field_name
        for _name in [n for n in unique_constraint_fields.split(",") if n]:
            validate_sql_identifier(_name)

        if key_is_not_string:
            Data = key_field_value
        else:
            Data = u(key_field_value)
        # All identifiers below (table, key_field_name, id_field_name, alternate_fields_names, unique_constraint_fields) are validated above
        if len(alternate_fields_names) > 0:
            if not use_alternate_fields_only:
                rs = _facade.safe_execute(
                    f"insert into {table} ({key_field_name} , {alternate_fields_names}) values ({Data},{alternate_fields_values}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(key_field_name, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
                )
            else:
                rs = _facade.safe_execute(
                    f"insert into {table} ({alternate_fields_names}) values ({alternate_fields_values}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(alternate_fields_names, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
                )
        else:
            rs = _facade.safe_execute(
                f"insert into {table} ({key_field_name}) values ({Data}) on conflict ({unique_constraint_fields}) do update set {make_set_excluded_clause(key_field_name, add_updated_at_timestamp)} returning {id_field_name}"  # nosec B608
            )

        the_id = rs[0][0]
        dict_enums[key_field_value] = the_id

        return the_id  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def _iter_partition_dates(from_date: date, to_date: date, partition_size: str):
    """Yield ``(period_start, period_end)`` date pairs stepping from ``from_date`` to
    ``to_date`` at ``partition_size`` granularity.

    2026-08-02 near-duplicate-function-body finding: create_postgres_range_partitions and
    delete_postgres_range_partitions independently duplicated this date-walk loop.
    """
    assert partition_size in ("day", "week", "month", "year")  # nosec B101 - partition_size only selects a relativedelta branch below, never spliced into SQL
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
        yield d, n
        d = n


def _partition_table_name(table_name: str, d: date) -> str:
    """The ``z_<table>_y<YYYY>m<MM>w<WW>d<DD>`` partition-name convention shared by
    create_postgres_range_partitions and delete_postgres_range_partitions -- both must agree on
    exactly this name or a create/delete pair silently target different tables."""
    return f"z_{table_name}_y{d.year:04d}m{d.month:02d}w{weekofmonth(d):02d}d{d.day:02d}"


def create_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str, bigint_degree: int = 0):
    """Creates one range partition of ``table_name`` per period between ``from_date`` and ``to_date``.

    ``partition_size`` is the period granularity (``day``/``week``/``month``/``year``) and each
    partition is named by the shared ``z_<table>_y..m..w..d..`` convention. Bounds are written as
    timestamps by default; a non-zero ``bigint_degree`` writes them instead as UTC epoch numbers scaled
    by ten to that power, for tables partitioned on an integer timestamp column.
    """
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    for d, n in _iter_partition_dates(from_date, to_date, partition_size):
        part_name = _partition_table_name(table_name, d)
        if bigint_degree is None or bigint_degree == 0:
            cmd = f"CREATE TABLE {part_name} PARTITION OF {table_name} FOR VALUES FROM ('{d:%Y-%m-%d %H:%M:%S}') TO ('{n:%Y-%m-%d %H:%M:%S}')"
        else:
            cmd = f"CREATE TABLE {part_name} PARTITION OF {table_name} FOR VALUES FROM ('{datetime_to_utc_timestamp(d)*int(10**bigint_degree)}') TO ('{datetime_to_utc_timestamp(n)*int(10**bigint_degree)}')"
        # print(cmd)
        _facade.safe_execute(cmd)


def delete_postgres_range_partitions(table_name: str, from_date: date, to_date: date, partition_size: str) -> None:
    """Drops the range partitions of ``table_name`` covering every period between ``from_date`` and
    ``to_date``, using the same ``partition_size`` granularity and naming convention that
    ``create_postgres_range_partitions`` used to create them."""
    # Validate table name to prevent SQL injection
    validate_sql_identifier(table_name)
    for d, _n in _iter_partition_dates(from_date, to_date, partition_size):
        cmd = f"drop table {_partition_table_name(table_name, d)}"
        _facade.safe_execute(cmd)


def enable_tables_sizes_approximation():
    """
    Example usage after the function is created:

    select table_name,nrows
        FROM
            cozmee.get_approximate_tables_sizes(
                array['public.amazon_products','public.amazon_users_reviews','public.amazon_products_reviews']
                )

    """

    _facade.safe_execute("""

CREATE or replace FUNCTION get_approximate_tables_sizes(tables_names text[])
RETURNS  TABLE (table_name text,nrows bigint)  AS
$func$

	SELECT relname::text,reltuples::bigint AS estimate FROM pg_class WHERE  oid in (select unnest($1)::regclass)
$func$ LANGUAGE sql;
""")


def create_enum_from_table(enum_name: str, table_name: str, id_field_name: str, desc_field_name: str) -> object:
    """Uses values and ids from a PG database to fill in Python enum.

    LinksTypes=create_enum_from_table("LinksTypes",table_name="links_types",id_field_name="id",desc_field_name="name")
    """
    dct = {}
    for id_field, desc_field in _facade.safe_execute(
        sql.SQL("select {id_field},{desc_field} from {table}").format(
            table=sql.Identifier(table_name), id_field=sql.Identifier(id_field_name), desc_field=sql.Identifier(desc_field_name)
        )
    ):
        dct[desc_field.upper()] = id_field
    return Enum(enum_name, dct)


def suggest_json_optimization(table: str, table_field: str, path: str = "", fields: Optional[list] = None, min_occurence_percent: float = 0.5, max_vals: int = 5) -> dict:
    """
    Aim is to remove lengthy fields that mostly holds a default value from the JSON.
    Absence of some JSON field is interpreted as null in PostGres.
    Result will be an optimization of an existing table.
    """
    if fields is None:
        fields = []

    # Validate identifiers to prevent SQL injection
    validate_sql_identifier(table)
    validate_sql_identifier(table_field)

    # Кандидат должен быть лидером по встречаемости. Также должен покрывать как минимум 90% вариантов. Кандидат не должн быть не-None, если поле имеет None в уже существующих значениях:
    # minHoursWeek [(None, 1244198), ('0', 150675)] - OK
    # risingTalent [(None, 97558), ('true', 441321), ('false', 856003)] - FAIL. false мог бы быть хорошим кандидатом, но уже встречается None.

    res = {}

    for field in fields:
        # Validate the JSON field key: it is spliced (quoted) directly into the query below
        validate_sql_identifier(field)

        # Prepare params
        if path == "" or path is None:
            full_path = table_field
        else:
            # `path` is spliced verbatim into the query below; restrict it to a JSON-navigation
            # fragment so it cannot close the surrounding expression.
            if not _facade._JSON_PATH_RE.match(path):
                raise ValueError(f"Invalid JSON navigation path: {path!r}")
            full_path = table_field + "->" + path

        # Ask DB
        # table/table_field/field/path all validated above (field is validated even though quoted, since it's spliced
        # directly; path is validated against _JSON_PATH_RE, which is the only grammar it may take)
        vals = _facade.safe_execute(f"""
                    select {full_path}->>'{field}' as val,count(*) as qty from {table} where {full_path} is not null group by {full_path}->>'{field}' order by qty desc
            """)  # nosec B608
        if not vals:  # safe_execute can return None/empty; nothing to analyse for this field
            continue

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
            logger.info("%s %s", field, vals)
        else:
            logger.info("%s %s ...", field, vals[:max_vals])

        # Print opt suggestions for that field
        if total_occurences > 0 and max_occurences / total_occurences >= min_occurence_percent:
            if not (has_none and best_val is not None):
                logger.info(
                    "Suggestion: make %s as default for %s. That will save %.2f%% of space.",
                    best_val,
                    field,
                    max_occurences / total_occurences * 100,
                )
                res[field] = best_val

    return res
