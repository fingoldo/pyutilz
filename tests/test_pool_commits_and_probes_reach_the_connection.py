"""`managed_connection` returns None, and so does a version that discards every write.

The context manager commits when its body leaves cleanly, then hands the connection back to the
pool. `release_connection` rolls back unconditionally on the way in -- which is the abort semantics
the exception path wants, and which means the commit is the ONLY thing separating a successful block
from one whose writes are thrown away. The module's own docstring records that: "without it the
unconditional rollback in release_connection silently discarded every write made through this
context manager."

Nothing about that is observable from the outside. The block returns normally either way, the
connection is returned to the pool either way, and the caller sees no error -- the rows are simply
not there the next time anyone looks.

`_ensure_connection` is the same shape in miniature: it proves a pooled connection is alive by
running `SELECT 1`, and returns None. Delete the query and it returns None faster, for every
connection, including the dead ones it exists to catch -- after which `get_connection` hands out a
broken connection and the failure surfaces in the caller's own query.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import psycopg2

from pyutilz.database import psycopg2_pool


@pytest.fixture
def pooled(monkeypatch):
    """A connection whose cursor records statements, wired in place of the real pool.

    The pool itself is replaced rather than exercised: what these tests are about is which calls
    reach the CONNECTION, and a real pool would only add a second object to keep in sync.
    """
    cursor = MagicMock(name="cursor")
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock(name="conn")
    conn.cursor = MagicMock(return_value=cursor)
    conn.closed = 0

    released: list = []
    monkeypatch.setattr(psycopg2_pool, "get_connection", lambda dsn, pool_max=8: conn)
    monkeypatch.setattr(psycopg2_pool, "release_connection", lambda c: released.append(c))
    conn._cursor = cursor
    conn._released = released
    return conn


def _statements(conn) -> list[str]:
    return [str(call.args[0]) for call in conn._cursor.execute.call_args_list if call.args]


class TestTheManagedBlockCommitsWhatItsBodyWrote:
    """Kills: `conn.commit()` -> `pass`, after which every write in the block is rolled back."""

    def test_a_clean_block_commits(self, pooled):
        with psycopg2_pool.managed_connection("dsn") as conn:
            conn.cursor().execute("INSERT INTO t VALUES (1)")

        pooled.commit.assert_called_once()

    def test_the_body_s_statement_actually_reached_the_connection(self, pooled):
        """Not merely "commit was called": a block that committed an empty transaction would
        satisfy the assertion above and write nothing."""
        with psycopg2_pool.managed_connection("dsn") as conn:
            conn.cursor().execute("INSERT INTO t VALUES (1)")

        assert "INSERT INTO t VALUES (1)" in _statements(pooled)

    def test_an_exception_does_not_commit(self, pooled):
        """The abort half. `release_connection` rolls back unconditionally, so NOT committing is
        what makes the block atomic -- committing here would persist a half-finished unit."""
        with pytest.raises(RuntimeError):
            with psycopg2_pool.managed_connection("dsn") as conn:
                conn.cursor().execute("INSERT INTO t VALUES (1)")
                raise RuntimeError("body failed")

        pooled.commit.assert_not_called()

    def test_the_connection_goes_back_to_the_pool_either_way(self, pooled):
        """A leaked connection is not an error either -- it is a pool that runs out later, in an
        unrelated caller."""
        with psycopg2_pool.managed_connection("dsn"):
            pass
        with pytest.raises(RuntimeError):
            with psycopg2_pool.managed_connection("dsn"):
                raise RuntimeError("body failed")

        assert len(pooled._released) == 2

    def test_the_commit_happens_before_the_release(self, pooled):
        """`release_connection` rolls back on the way in. Commit after it and the rollback wins,
        which is precisely the bug the ordering exists to prevent."""
        order: list[str] = []
        pooled.commit.side_effect = lambda: order.append("commit")

        def _release(conn):
            order.append("release")

        monkey = pytest.MonkeyPatch()
        monkey.setattr(psycopg2_pool, "release_connection", _release)
        try:
            with psycopg2_pool.managed_connection("dsn"):
                pass
        finally:
            monkey.undo()

        assert order == ["commit", "release"]


class TestTheLivenessProbeActuallyProbes:
    """`_ensure_connection` returns None whether it asked the database anything or not."""

    def test_it_runs_a_query(self, pooled):
        psycopg2_pool._ensure_connection(pooled)

        assert "SELECT 1" in _statements(pooled)

    def test_a_dead_connection_is_rolled_back_and_re_raised(self, pooled):
        """The direction that matters. A probe that cannot fail hands out broken connections, and
        the error then surfaces inside the caller's own query, naming none of this."""
        pooled._cursor.execute.side_effect = psycopg2.OperationalError("server closed the connection")

        with pytest.raises(psycopg2.Error):
            psycopg2_pool._ensure_connection(pooled)

        pooled.rollback.assert_called()

    def test_a_healthy_connection_is_not_rolled_back(self, pooled):
        """The other side of the guard: probing must not disturb an in-flight transaction."""
        psycopg2_pool._ensure_connection(pooled)

        pooled.rollback.assert_not_called()
