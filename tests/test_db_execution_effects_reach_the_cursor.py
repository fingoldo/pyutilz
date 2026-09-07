"""`execute_alchemy` returns None, and so does a version whose statement never reached the server.

It is the raw-SQL escape hatch: the caller hands it a statement, it opens a connection, executes and
commits, and returns None. SQLAlchemy 2.0 does not autocommit, so dropping the commit does not raise
-- the transaction is discarded when the connection closes and the function still returns None. For
a function whose whole purpose is running DDL and maintenance statements, that is a migration that
reports success and changed nothing.

`basic_db_execute` is the parameterised path, and its own effect is a DISPATCH: the same call routes
to `cur.execute(statement, data)` or to psycopg2's `execute_values(cur, statement, data)` depending
on `ex_type`. Both return rows, so picking the wrong one is not visible in the return value -- it is
visible in how long a bulk insert takes, and in nothing else.

The retry loop is asserted from both ends because it is the part that can silently do nothing: a
budget that never reaches the statement, and a permanently-broken statement retried as if transient.
Both are shapes this module has already had to fix.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyutilz.database.db import execution


@pytest.fixture
def alchemy(monkeypatch):
    """A SQLAlchemy connection recording what was executed and committed."""
    connection = MagicMock(name="connection")
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)

    engine = MagicMock(name="engine")
    engine.connect = MagicMock(return_value=connection)

    monkeypatch.setattr(execution._facade, "conn_alchemy", engine)
    monkeypatch.setattr(execution._facade, "sleep", lambda _s: None)
    engine._connection = connection
    return engine


def _statements(engine) -> list[str]:
    return [str(call.args[0]) for call in engine._connection.execute.call_args_list if call.args]


class TestTheRawEscapeHatchCommits:
    """Kills: `connection.commit()` -> `pass`, which returns None exactly as success does."""

    def test_the_statement_reaches_the_connection(self, alchemy):
        execution.execute_alchemy("CREATE INDEX CONCURRENTLY idx ON t (a)")

        assert "CREATE INDEX CONCURRENTLY idx ON t (a)" in _statements(alchemy)

    def test_it_commits(self, alchemy):
        """SQLAlchemy 2.0 does not autocommit, so without this the statement is discarded when the
        connection closes -- and the function still returns None."""
        execution.execute_alchemy("CREATE INDEX idx ON t (a)")

        alchemy._connection.commit.assert_called_once()

    def test_the_commit_is_inside_the_connection_block(self, alchemy):
        """Committing after the block would commit nothing: the connection is already returned."""
        order: list[str] = []
        alchemy._connection.commit.side_effect = lambda: order.append("commit")
        alchemy._connection.__exit__ = MagicMock(side_effect=lambda *a: order.append("close") or False)

        execution.execute_alchemy("CREATE INDEX idx ON t (a)")

        assert order == ["commit", "close"]

    def test_a_transient_failure_is_retried(self, alchemy):
        """The retry is why this returns None rather than raising on the first blip."""
        attempts = {"n": 0}

        def _flaky(*_a, **_k):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("server closed the connection unexpectedly")
            return MagicMock()

        alchemy._connection.execute.side_effect = _flaky

        execution.execute_alchemy("CREATE INDEX idx ON t (a)", max_retries=3)

        assert attempts["n"] == 2
        alchemy._connection.commit.assert_called_once()

    def test_a_permanently_broken_statement_raises_rather_than_returning(self, alchemy):
        """The regression the module's own comment records: it used to fall off the end once the
        budget ran out, so a syntax error or a permission denial returned None -- indistinguishable
        from a statement that ran."""
        alchemy._connection.execute.side_effect = RuntimeError('syntax error at or near "CREAT"')

        with pytest.raises(RuntimeError, match="syntax error"):
            execution.execute_alchemy("CREAT INDEX idx ON t (a)", max_retries=2)

    def test_a_non_positive_budget_is_refused_rather_than_silently_skipping(self, alchemy):
        """With `max_retries=0` the loop body never ran, so the statement was never sent and the
        function returned None after logging that it gave up."""
        with pytest.raises(ValueError, match="max_retries"):
            execution.execute_alchemy("CREATE INDEX idx ON t (a)", max_retries=0)

        alchemy._connection.execute.assert_not_called()


@pytest.fixture
def psycopg(monkeypatch):
    """A psycopg2-style cursor, plus a recorder standing in for `execute_values`."""
    cursor = MagicMock(name="cursor")
    cursor.description = None
    cursor.fetchall = MagicMock(return_value=[])
    monkeypatch.setattr(execution._facade, "get_cursor", lambda **_k: cursor)
    monkeypatch.setattr(execution._facade, "sleep", lambda _s: None)

    # A MagicMock rather than a recording lambda: the assertions below then inspect the CALL to
    # psycopg2's bulk helper directly, which is the effect under test, instead of a list a closure
    # happened to append to.
    # Patched ON the module, and asserted through it below as `execution.execute_values`: the
    # binding the code under test actually resolves is the one worth inspecting, and reaching it
    # through a local alias hides which symbol the assertion is about.
    monkeypatch.setattr(execution, "execute_values", MagicMock(name="execute_values"))
    return cursor


class TestTheDispatchPicksTheRightDriverCall:
    """`ex_type` chooses between two effects, and the return value cannot tell them apart."""

    def test_execute_goes_to_the_cursor(self, psycopg):
        execution.basic_db_execute("execute", "INSERT INTO t VALUES (%s)", data=(1,))

        assert psycopg.execute.call_args.args[0] == "INSERT INTO t VALUES (%s)"
        execution.execute_values.assert_not_called()

    def test_execute_values_goes_to_the_bulk_helper(self, psycopg):
        """Routing a bulk insert through plain `execute` is correct AND one round trip per row --
        invisible in the result, visible only in how long the load takes."""
        rows = [(1,), (2,), (3,)]

        execution.basic_db_execute("execute_values", "INSERT INTO t VALUES %s", data=rows)

        execution.execute_values.assert_called_once()
        assert execution.execute_values.call_args.args[1:3] == ("INSERT INTO t VALUES %s", rows)
        psycopg.execute.assert_not_called()

    def test_the_caller_s_data_is_passed_through(self, psycopg):
        """A statement executed with the wrong bindings still returns rows."""
        execution.basic_db_execute("execute", "SELECT * FROM t WHERE a = %s", data=(7,))

        assert psycopg.execute.call_args.args[1] == (7,)
