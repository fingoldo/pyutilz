import pytest
import ast


def test_f08_basic_db_execute_has_no_implicit_none_terminal():
    """Its retry loop can fall through (named-cursor collisions exhausting the budget); the terminal
    is an explicit raise naming the budget, not an implicit None a caller reads as "no rows"."""
    import inspect

    from pyutilz.database.db import execution

    tree = ast.parse(inspect.getsource(execution))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "basic_db_execute")
    assert isinstance(func.body[-1], ast.Raise)


def test_f04_get_table_fields_never_returns_none(monkeypatch):
    """Its result is concatenated into SQL, so None (or the literal text "None") must be impossible."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.execution import get_table_fields

    class _Cur:
        description = None

        def execute(self, *a, **k):
            return None

        def fetchall(self):
            return []

    monkeypatch.setattr(facade, "get_cursor_type", lambda *a, **k: "c")
    monkeypatch.setattr(facade, "get_cursor", lambda *a, **k: _Cur())
    with pytest.raises(RuntimeError, match="no cursor description"):
        get_table_fields("orders", "o")


def test_f08_basic_db_execute_returns_an_empty_list_for_a_no_result_statement(monkeypatch):
    """The docstring promises an empty list for a statement with no result set; a DuplicateTable
    collision (a CREATE that already exists) is exactly that case and used to return a bare None,
    so `for row in basic_db_execute(...)` raised TypeError on a path the documentation handles."""
    from psycopg2.errors import DuplicateTable

    import pyutilz.database.db as facade
    from pyutilz.database.db.execution import basic_db_execute

    class _Cur:
        def execute(self, *a, **k):
            raise DuplicateTable("relation already exists")

    monkeypatch.setattr(facade, "get_cursor_type", lambda *a, **k: "c")
    monkeypatch.setattr(facade, "get_cursor", lambda *a, **k: _Cur())
    assert basic_db_execute("execute", "create table t (id int)") == []
