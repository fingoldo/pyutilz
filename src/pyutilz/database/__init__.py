"""PyUtilz database subpackage."""

# Eagerly re-exported (stdlib-only module, no optional-dep cost) so the domain's typed errors are
# as discoverable as pyutilz.llm's: `from pyutilz.database import DatabaseConnectionError` works.
from .exceptions import DatabaseError, DatabaseConnectionError, SQLValidationError

__all__ = [
    "db",
    "redislib",
    "deltalakes",
    "psycopg2_pool",
    "exceptions",
    "DatabaseError",
    "DatabaseConnectionError",
    "SQLValidationError",
]
