"""Pure, stateless SQL/identifier helpers (quoting, identifier validation, ON CONFLICT clause building) shared by db.py."""

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

from pyutilz.database.exceptions import SQLValidationError

# ----------------------------------------------------------------------------------------------------------------------------
# SQL Injection Protection
# ----------------------------------------------------------------------------------------------------------------------------

import re

# Pre-compiled at module level: validate_sql_identifier is on a hot query-building path (called per identifier per query).
# timeit micro-benchmark (1M iters, "my_table_name"): re.match(pattern_str, s) ~0.82us/call
# vs _SQL_IDENTIFIER_RE.match(s) ~0.36us/call -> ~2.26x faster by avoiding the per-call pattern-cache lookup.
# \Z (not $): $ matches both the true end of string AND immediately before a trailing "\n" in
# Python's re (without re.MULTILINE) -- e.g. pat.match("users\n") incorrectly succeeds with $.
# Found 2026-07-21 audit: an identifier with a trailing newline reached SQL text silently
# accepted where it should have been rejected. \Z matches only the absolute end of the string.
_SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\Z")


def validate_sql_identifier(identifier: Optional[str]) -> str:
    """Validate that an identifier (table name, column name) is safe to use in SQL.

    Raises ValueError if the identifier contains potentially malicious characters.
    Valid identifiers must match: alphanumeric, underscore, start with letter/underscore.
    """
    if not isinstance(identifier, str):
        raise SQLValidationError(f"SQL identifier must be a string, got {type(identifier)}")
    if not _SQL_IDENTIFIER_RE.match(identifier):
        raise SQLValidationError(f"Invalid SQL identifier: {identifier!r}. Must contain only alphanumeric and underscore, start with letter or underscore.")
    return identifier


def validate_sql_qualified_identifier(identifier: str) -> str:
    """Validate a possibly schema-qualified SQL identifier, e.g. ``schema.table``.

    Each dot-separated part is validated individually with ``validate_sql_identifier``.
    Raises ValueError if the identifier (or any of its parts) is invalid.
    """
    if not isinstance(identifier, str):
        raise SQLValidationError(f"SQL identifier must be a string, got {type(identifier)}")
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
        elif mode in ("select", "update"):
            # Regression fix: "update" fell through with no branch at all (only "insert"/
            # "select" were handled), so `templates` stayed empty for update-mode callers while
            # `values` was still filled correctly -- db_command(mode="update", ...) always built
            # syntactically-invalid SQL ("update  mytable set  where ") with mismatched
            # parameter counts. "update" needs the identical "field=%s" shape as "select" (both
            # build a `field=value` fragment; only "insert" wants a bare positional "%s").
            templates.append(key + "=%s")
    return values, templates


def u(str_val, symb="'"):
    """
    Quotes str_val as a SQL string literal with symb (doubling any embedded symb to escape it), or returns "null" if str_val is None.

    Assumes standard_conforming_strings=on (Postgres default since 9.1) / NO_BACKSLASH_ESCAPES=on
    (MySQL): a trailing backslash is NOT separately escaped, only doubled-quote escaping is
    applied. Under a backslash-escaping dialect (NO_BACKSLASH_ESCAPES=off, or pre-9.1 Postgres
    with standard_conforming_strings=off) a trailing backslash would escape the closing quote
    instead of terminating the literal -- prefer parameterized queries over u()/nu() wherever
    the caller can, especially against MySQL (this module explicitly supports both flavors, see
    db/__init__.py's db_flavor assert).
    """
    if str_val is None:
        return "null"
    if not isinstance(str_val, str):
        raise TypeError(f"u() requires a str or None, got {type(str_val)}")
    return symb + str_val.replace(symb, symb * 2) + symb


def nu(str_val, symb="'"):
    """
    Like u(), but also returns "null" for an empty string (not just None).
    """
    if str_val is None or (isinstance(str_val, str) and len(str_val) == 0):
        return "null"
    else:
        return u(str_val, symb)


def MakeSetExcludedClause(sFields: str, bAddUpdatedAtTimestamp: Optional[str] = None) -> str:
    """
    Builds an "ON CONFLICT DO UPDATE SET" body from a comma-separated field list sFields, mapping each
    field to excluded.field. If bAddUpdatedAtTimestamp is given, appends "{bAddUpdatedAtTimestamp}=(now() at time zone 'utc')"
    as the final clause instead of a trailing comma.
    """

    res = ""
    v = sFields.split(",")
    for i in v:
        # Validate field name to prevent SQL injection
        validate_sql_identifier(i)
        res = res + i + "=excluded." + i + ","
    if bAddUpdatedAtTimestamp:
        # Regression fix (SQL injection): every OTHER field name in this function is validated
        # via validate_sql_identifier() before being spliced into the generated SQL --
        # bAddUpdatedAtTimestamp was the one exception, spliced raw with no validation and no
        # "accepted raw fragment" docstring warning either (unlike update_if_now()'s `clause`
        # param a few lines below, which IS explicitly documented as an accepted raw fragment).
        # Reachable from the public GetIdByKeyFieldAndInsertIfNeeded() in db/__init__.py.
        validate_sql_identifier(bAddUpdatedAtTimestamp)
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
