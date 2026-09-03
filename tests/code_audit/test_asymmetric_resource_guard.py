"""Scanner tests for asymmetric_resource_guard, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_asymmetric_resource_guard,
)

from ._helpers import _write

# ---- asymmetric_resource_guard --------------------------------------------


def test_asymmetric_resource_guard_transaction_flagged(tmp_path: Path):
    """The motivating shape: query_rows() correctly wraps conn.cursor() in a
    transaction; prefetch_resume_cache(), a sibling method of the SAME class,
    performs the identical conn.cursor() call unwrapped."""
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def query_rows(self, conn, sql, params):
        async with conn.transaction():
            cur = conn.cursor(sql, *params)
            return [row async for row in cur]

    async def prefetch_resume_cache(self, conn, sql, params):
        cur = conn.cursor(sql, *params)
        return [row async for row in cur]
""",
    )
    findings = scan_asymmetric_resource_guard(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "asymmetric_resource_guard"
    assert findings[0].severity == "P0"
    assert "conn.cursor" in findings[0].detail
    assert "prefetch_resume_cache" in findings[0].detail
    assert "query_rows" in findings[0].detail


def test_asymmetric_resource_guard_lock_shape_flagged(tmp_path: Path):
    """Bare self._lock context-manager guard shape (not a .transaction() call)."""
    _write(
        tmp_path,
        "storage.py",
        """
class FileStorage:
    async def close(self):
        self._db.execute("PRAGMA optimize")

    async def write(self, row):
        async with self._lock:
            self._db.execute("insert ...")
""",
    )
    findings = scan_asymmetric_resource_guard(tmp_path)
    assert len(findings) == 1, findings
    assert "self._db.execute" in findings[0].detail
    assert "close" in findings[0].detail
    assert "write" in findings[0].detail


def test_asymmetric_resource_guard_consistently_guarded_is_clean(tmp_path: Path):
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def query_rows(self, conn, sql, params):
        async with conn.transaction():
            return conn.cursor(sql, *params)

    async def prefetch_resume_cache(self, conn, sql, params):
        async with conn.transaction():
            return conn.cursor(sql, *params)
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_consistently_unguarded_is_clean(tmp_path: Path):
    """Both methods agree on NOT guarding -- no asymmetry, nothing to flag
    (this scanner only fires when one method demonstrates the correct
    pattern and a sibling doesn't; it never invents a rule from nothing)."""
    _write(
        tmp_path,
        "storage.py",
        """
class PostgresStorage:
    async def a(self, conn):
        return conn.execute("select 1")

    async def b(self, conn):
        return conn.execute("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_different_classes_not_compared(tmp_path: Path):
    """The same operation-shape guarded in one class and unguarded in an
    UNRELATED class is not a finding -- the whole point is that ONE class's
    own code already demonstrates its own correct pattern."""
    _write(
        tmp_path,
        "storage.py",
        """
class A:
    async def guarded(self, conn):
        async with conn.transaction():
            return conn.cursor("select 1")

class B:
    async def unguarded(self, conn):
        return conn.cursor("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_single_method_never_flagged(tmp_path: Path):
    """A class with only one method touching a given operation-shape has no
    sibling to compare against -- can't be asymmetric by definition."""
    _write(
        tmp_path,
        "storage.py",
        """
class Solo:
    async def only(self, conn):
        return conn.cursor("select 1")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []


def test_asymmetric_resource_guard_custom_guard_names(tmp_path: Path):
    _write(
        tmp_path,
        "storage.py",
        """
class Store:
    async def a(self, conn):
        async with conn.my_custom_guard():
            return conn.execute("select 1")

    async def b(self, conn):
        return conn.execute("select 2")
""",
    )
    assert scan_asymmetric_resource_guard(tmp_path) == []
    findings = scan_asymmetric_resource_guard(tmp_path, guard_call_names=frozenset({"my_custom_guard"}))
    assert len(findings) == 1
