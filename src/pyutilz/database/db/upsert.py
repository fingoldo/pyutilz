# ----------------------------------------------------------------------------------------------------------------------------
# Pure upsert / history query builder carved out of db.py.
# build_upsert_query constructs a SQL string; it does NOT touch the
# module-level connection globals. Re-exported by the db.__init__ facade.
# ----------------------------------------------------------------------------------------------------------------------------


def build_upsert_query(
    fields_names: list,
    table_name: str,
    conflict_fields: list = None,
    fields_types: dict = None,
    skip_fields: list = None,
    timestamp_check_fields: list = None,
    timestamp_update_fields: list = None,
    on_conflict_update_fields: list = None,
    on_conflict_update_values: dict = None,
    custom_onconflict: str = None,
    returning_fields: list = None,
    history_table_name: str = None,
    history_fields: list = None,
    history_fields_aliases: dict = None,
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

    if history_table_name and len(history_table_name) > 0:
        assert len(conflict_fields) > 0

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
            join_condtion = None
            hist_query += f" returning {','.join(f_)}"

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
                    the_join_condtion = "where {join_condtion}"
                else:
                    the_join_condtion = ""

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
