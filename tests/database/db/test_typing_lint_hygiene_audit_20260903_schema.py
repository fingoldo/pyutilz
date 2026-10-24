import pytest


def test_f03_check_if_pg_table_exists_reports_unknown_as_none(monkeypatch):
    """A falsy safe_execute result is "I could not find out", explicitly None rather than a fall-off."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.schema import check_if_pg_table_exists

    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: [])
    assert check_if_pg_table_exists("t") is None
    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: [[True]])
    assert check_if_pg_table_exists("t") is True


def test_f03_ensure_pg_table_exists_refuses_to_create_on_an_unknown_probe(monkeypatch):
    """ "Unknown" must not be read as "absent" and followed by a CREATE against a database that could not be queried."""
    import pyutilz.database.db as facade
    from pyutilz.database.db.schema import ensure_pg_table_exists

    monkeypatch.setattr(facade, "check_if_pg_table_exists", lambda *a, **k: None)
    monkeypatch.setattr(facade, "safe_execute", lambda *a, **k: pytest.fail("CREATE must not run on an unknown probe"))
    with pytest.raises(RuntimeError, match="Could not determine whether table"):
        ensure_pg_table_exists("mytable")
