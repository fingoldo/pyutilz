# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Pure, stateless SQL / identifier helpers carved out of db.py.
# These functions do NOT touch the module-level connection globals
# (conn/cur/cursors) and are safe to live in their own submodule; the
# db.__init__ facade re-exports every public name below.
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Optional
import json

# ----------------------------------------------------------------------------------------------------------------------------
# SQL Injection Protection
# ----------------------------------------------------------------------------------------------------------------------------

import re

# Pre-compiled at module level: validate_sql_identifier is on a hot query-building path (called per identifier per query).
# timeit micro-benchmark (1M iters, "my_table_name"): re.match(pattern_str, s) ~0.82us/call
# vs _SQL_IDENTIFIER_RE.match(s) ~0.36us/call -> ~2.26x faster by avoiding the per-call pattern-cache lookup.
_SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_sql_identifier(identifier: str) -> str:
    """Validate that an identifier (table name, column name) is safe to use in SQL.

    Raises ValueError if the identifier contains potentially malicious characters.
    Valid identifiers must match: alphanumeric, underscore, start with letter/underscore.
    """
    if not isinstance(identifier, str):
        raise ValueError(f"SQL identifier must be a string, got {type(identifier)}")
    if not _SQL_IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}. Must contain only alphanumeric and underscore, start with letter or underscore.")
    return identifier


def validate_sql_qualified_identifier(identifier: str) -> str:
    """Validate a possibly schema-qualified SQL identifier, e.g. ``schema.table``.

    Each dot-separated part is validated individually with ``validate_sql_identifier``.
    Raises ValueError if the identifier (or any of its parts) is invalid.
    """
    if not isinstance(identifier, str):
        raise ValueError(f"SQL identifier must be a string, got {type(identifier)}")
    for part in identifier.split("."):
        validate_sql_identifier(part)
    return identifier


def construct_templates_and_values(mode, fields, replace_values, source, jsonize):
    """
    Helper sub to assist filling correct templates in db_command sub, based on mode passed.
    """
    values, templates = [], []
    for key in fields:
        # Validate field name to prevent SQL injection (key is spliced into the template below)
        validate_sql_identifier(key)
        if key in replace_values:
            value = replace_values[key]
        else:
            value = source.get(key, None)

        if jsonize:
            if type(value) in (dict, list):
                # Sorted keys give stable serialization (stable hashing/dedup rule). orjson is ~5-10x faster than
                # stdlib json on dumps; falls back to stdlib only if orjson missing (kept optional -- core has no
                # hard requirements).
                try:
                    import orjson  # type: ignore

                    value = orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode("utf-8")
                except ImportError:
                    # separators matches orjson's compact (no-space) output so the
                    # stored/hashed string is stable regardless of which backend ran.
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))

        values.append(value)
        if mode == "insert":
            templates.append("%s")
        elif mode == "select":
            templates.append(key + "=%s")
    return values, templates


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


def MakeSetExcludedClause(sFields: str, bAddUpdatedAtTimestamp: Optional[str] = None) -> str:

    res = ""
    v = sFields.split(",")
    for i in v:
        # Validate field name to prevent SQL injection
        validate_sql_identifier(i)
        res = res + i + "=excluded." + i + ","
    if bAddUpdatedAtTimestamp:
        res = res + f"{bAddUpdatedAtTimestamp}=(now() at time zone 'utc')"  # updated_at
    else:
        res = res[:-1]
    return res


def update_if_now(added_at: str, clause: str) -> str:
    """
    builds an onconflict_clause. Updates existing data only if now is specified in added_at.
    """
    if "now" in added_at.lower():
        # `clause` is an accepted raw SQL fragment by design (a full SET clause built by the caller), not an identifier
        onconflict_clause = f"""
            do update set
                {clause}
        """  # nosec B608
    else:
        onconflict_clause = """do nothing"""

    return onconflict_clause
