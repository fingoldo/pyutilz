"""Pure upsert / history query builder carved out of db.py.

``build_upsert_query`` constructs a SQL string; it does NOT touch the
module-level connection globals. Re-exported by the db.__init__ facade.
"""
# ----------------------------------------------------------------------------------------------------------------------------
# Pure upsert / history query builder carved out of db.py.
# build_upsert_query constructs a SQL string; it does NOT touch the
# module-level connection globals. Re-exported by the db.__init__ facade.
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.database.db.sql_helpers import validate_sql_identifier
from typing import Iterable, Optional


def build_upsert_query(
    fields_names: list,
    table_name: str,
    conflict_fields: Optional[list] = None,
    fields_types: Optional[dict] = None,
    skip_fields: Optional[list] = None,
    timestamp_check_fields: Optional[list] = None,
    timestamp_update_fields: Optional[list] = None,
    on_conflict_update_fields: Optional[list] = None,
    on_conflict_update_values: Optional[dict] = None,
    custom_onconflict: Optional[str] = None,
    returning_fields: Optional[list] = None,
    history_table_name: Optional[str] = None,
    history_fields: Optional[list] = None,
    history_fields_aliases: Optional[dict] = None,
    hash_fields: Optional[Iterable[str]] = "",
    default_timestamp: str = "now()",
) -> str:
    """
    Inserts new records or Updates fields in a Postgres table
    Optionally saves data into History table after comparing Hash field of the new and old record.

    Params:
    timestamp_check_fields: when element was last checked (queried from source)
    timestamp_update_fields: when element was last updated (queried from source and found changed)
    """
    if conflict_fields is None:
        conflict_fields = ["id"]
    if fields_types is None:
        fields_types = {}
    if history_fields is None:
        history_fields = []
    if history_fields_aliases is None:
        history_fields_aliases = {}
    if on_conflict_update_fields is None:
        on_conflict_update_fields = []
    if on_conflict_update_values is None:
        on_conflict_update_values = {}
    if returning_fields is None:
        returning_fields = []
    if skip_fields is None:
        skip_fields = []
    if timestamp_check_fields is None:
        timestamp_check_fields = []
    if timestamp_update_fields is None:
        timestamp_update_fields = []
    if isinstance(hash_fields, str):
        # A bare str would otherwise be iterated character-by-character below (for hash_field in
        # hash_fields), silently corrupting the generated SQL. "" (the default/no-op) stays as-is;
        # any other bare str is treated as a single field name, matching the pre-fix behavior most
        # callers actually relied on when they passed a single hash column.
        hash_fields = [hash_fields] if hash_fields else []
    else:
        hash_fields = list(hash_fields) if hash_fields is not None else []
    # ------------------------------
    # Checks!
    # ------------------------------

    assert len(fields_names) > 0  # nosec B101 - trivial arity guard against building a query with zero columns, not an identifier check

    # Validate every identifier that ends up spliced into the raw SQL below (table/history table/column names).
    # ``custom_onconflict`` and ``on_conflict_update_values`` values are accepted as raw SQL fragments by design.
    validate_sql_identifier(table_name)
    if history_table_name:
        validate_sql_identifier(history_table_name)
    for _identifier_list in (
        fields_names,
        conflict_fields,
        on_conflict_update_fields,
        returning_fields,
        history_fields,
        timestamp_check_fields,
        timestamp_update_fields,
        skip_fields,
        list(history_fields_aliases.keys()),
        list(history_fields_aliases.values()),
    ):
        for _field in _identifier_list:
            validate_sql_identifier(_field)
    if hash_fields:
        for hash_field in hash_fields:
            validate_sql_identifier(hash_field)

    for field in timestamp_check_fields:
        assert field not in fields_names  # nosec B101 - both lists were already identifier-validated above; this only guards against duplicate columns in the generated SELECT list, not injection
        assert field not in fields_types  # nosec B101 - same duplicate-column guard as above; fields_types keys are matched against already-validated fields_names

    for field_name, field_type in fields_types.items():
        assert field_name in fields_names  # nosec B101 - field_name is only ever used via membership tests below (loop iterates actual_fields_names, not fields_types keys), so a stray key here cannot reach the generated SQL
        if field_type not in "int bigint smallint float real numeric json jsonb text timestamp".split(" "):
            # field_type is spliced verbatim into the SQL as "field::<field_type>" below (line ~119/146); under
            # `python -O` a skipped assert would let an unvalidated field_type inject arbitrary SQL, so this must
            # be a real, non-disable-able check rather than an assert.
            raise ValueError(f"Invalid field_type for {field_name!r}: {field_type!r}")

    rev_on_conflict_update_values = {}
    if len(on_conflict_update_values) > 0:
        for _field in on_conflict_update_values.keys():
            # assert _field in fields_names
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
    # table_name/actual_fields_names validated above (actual_fields_names is derived from fields_names/timestamp_check_fields,
    # minus skip_fields, so its members were already validated as identifiers)
    fresh_query = f"""
    insert into {table_name} ({','.join(actual_fields_names)})
    select {','.join(values)}
    from data """  # nosec B608

    # ------------------------------
    # On conflict
    # ------------------------------

    if custom_onconflict or (conflict_fields is not None and len(conflict_fields) > 0 and len(on_conflict_update_fields) > 0):
        if not custom_onconflict:
            for field in conflict_fields:
                assert field in fields_names  # nosec B101 - both lists were already identifier-validated above; this only enforces that ON CONFLICT targets are also insert columns, a business-logic invariant

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

            # conflict_fields validated above; update_values entries are "field=..." fragments built from validated
            # field names plus on_conflict_update_values entries, which are accepted raw SQL fragments by design
            fresh_query += f"on conflict ({','.join(conflict_fields)}) do update set {','.join(update_values)}"  # nosec B608
        else:
            fresh_query += custom_onconflict
    else:
        fresh_query += "on conflict do nothing "

    if len(history_fields) > 0:
        returning_fields = history_fields
        if hash_fields:
            for hash_field in hash_fields:
                assert hash_field in fields_names  # nosec B101 - both lists were already identifier-validated above; this only enforces hash_fields is a subset of the insert columns, a business-logic invariant
                if hash_field not in history_fields:
                    returning_fields += [hash_field]
        fresh_query += f" returning {','.join(returning_fields)}"

    query += ", fresh_data as (" + fresh_query + ")"

    if history_table_name and len(history_table_name) > 0:
        assert len(conflict_fields) > 0  # nosec B101 - arity guard: a history table needs a conflict key to join against, not an identifier check

        # history_fields validated above; history_fields_final entries come from history_fields_aliases (also validated) or
        # history_fields itself, so every element here is a validated identifier
        history_fields_final = [history_fields_aliases.get(field, field) for field in history_fields]
        hist_query = f"""
        ,changed_data as (insert into {history_table_name}({','.join(history_fields_final)}) select {','.join(['u.'+field for field in history_fields])} from fresh_data u
        """  # nosec B608
        if len(timestamp_update_fields) > 0:
            the_list = conflict_fields + timestamp_check_fields
        else:
            the_list = conflict_fields

        f_ = [field for field in the_list if field not in history_fields_aliases]
        if len(timestamp_update_fields) > 0:
            for field in timestamp_check_fields:
                new_field = f"{history_fields_aliases.get(field)} as {field}" if field in history_fields_aliases else field
                if new_field not in f_:
                    f_ = [*f_, new_field]

        if hash_fields:
            join_condtion = " and ".join([f"u.{field}=c.{field}" for field in conflict_fields])

            hash_changing_conds = [
                f"((c.{hash_field} is null and u.{hash_field} is not null) or (c.{hash_field} is not null and u.{hash_field} is null) or (c.{hash_field}<>u.{hash_field}))"
                for hash_field in hash_fields
            ]
            hash_changing_cond = " OR ".join(hash_changing_conds)

            hist_query += f"left join {table_name} c on {join_condtion} where ({hash_changing_cond}) returning {','.join(f_)}"
        else:
            join_condtion = None
            hist_query += f" returning {','.join(f_)}"

        hist_query += ")"
        query += hist_query

        if len(timestamp_update_fields) > 0:
            assert len(timestamp_update_fields) == len(timestamp_check_fields)  # nosec B101 - arity guard so the zip() below pairs fields 1:1; both lists were already identifier-validated above

            upd_fields_and_vals = ",".join(
                [f"{ufield}=c.{cfield}" for ufield, cfield in zip(timestamp_update_fields, timestamp_check_fields + timestamp_check_fields)]
            )

            the_join_condtion = f"where {join_condtion}" if join_condtion else ""

            # table_name/upd_fields_and_vals built from validated identifiers above
            query += f" select * from changed_data); with tmp as (update {table_name} AS u set {upd_fields_and_vals} from changed_data as c {the_join_condtion}) select count(*) from changed_data;"  # nosec B608

            query = "create temp table changed_data ON COMMIT DROP as (" + query
        else:
            query += " select 1 from changed_data"
    else:
        query += " select 1 from fresh_data"
    return query
