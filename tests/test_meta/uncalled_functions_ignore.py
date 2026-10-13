"""Classified exceptions for the py-ci-shared ``uncalled_functions`` gate.

Each entry is a bare function name the gate must not count, with the one-line reason it is allowed to stay
without a production call site. Only two categories are classified here: deprecated aliases kept for backward
compatibility, and PEP 562 module ``__dir__`` hooks invoked by ``dir()``. Plain public utility API is not
classified here yet; that category is pending an owner decision.
"""

from typing import Dict

UNCALLED_FUNCTIONS_IGNORE: Dict[str, str] = {
    "__dir__": "PEP 562 module hook invoked by dir(), not by a call site in the code.",
    "EnsurePgTableExists": "Deprecated alias superseded by ensure_pg_table_exists in database/db/schema.py.",
    "GetIdByKeyFieldAndInsertIfNeeded": "Deprecated alias superseded by get_id_by_key_field_and_insert_if_needed.",
    "ReadTableIntoDic": "Deprecated alias superseded by read_table_into_dict.",
    "ReadTableIntoDicReversed": "Deprecated alias superseded by read_table_into_dict_reversed.",
    "MakeSetExcludedClause": "Deprecated alias superseded by make_set_excluded_clause in database/db/sql_helpers.py.",
    "remove_stale_columns": "Deprecated alias superseded by get_non_stale_columns in data/pandaslib/frames.py.",
}
