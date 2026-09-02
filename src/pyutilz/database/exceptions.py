"""Typed exceptions for the database domain, mirroring pyutilz.llm.exceptions's pattern.

Lets callers discriminate error conditions via ``except SpecificError`` instead of string-matching
a generic ``RuntimeError``/``ValueError`` message (which breaks silently the moment the message
wording changes). Not a full hierarchy for every condition -- just the highest-traffic ones.
"""

from __future__ import annotations


class DatabaseError(Exception):
    """Root of the database domain's exception hierarchy.

    Exists so ``except DatabaseError`` reliably catches every error this package raises on
    purpose. Each concrete subclass keeps its original builtin base as a SECOND base, so code
    written against ``except RuntimeError``/``except ValueError`` keeps working unchanged.
    """


class DatabaseConnectionError(DatabaseError, RuntimeError):
    """A required database/cache connection has not been established yet.

    Distinct from a genuine transport-level connection failure (e.g. ``redis.exceptions.
    ConnectionError``): this means the caller forgot to call the connect step first, not that
    the connection attempt itself failed.
    """


class SQLValidationError(DatabaseError, ValueError):
    """A SQL identifier, field name, or query-construction argument failed validation.

    Raised by ``validate_sql_identifier``/``validate_sql_qualified_identifier`` and callers that
    reject caller-supplied values before splicing them into raw SQL.
    """
