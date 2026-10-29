"""Behaviour of ``pyutilz.database.db.connection``: connection parameters carry the database, host, port and credentials,
and add the SSL mode and the schema search path only when those are given; the cursor cache key is the cursor class name,
suffixed for a named (server-side) cursor; and only ``postgres`` and ``mysql`` are accepted as a database flavour. Nothing
here opens a connection."""

from __future__ import annotations

import pytest

from pyutilz.database.db import connection

_FAKE_PASSWORD = "not-a-real-password"  # pragma: allowlist secret


class TestConnectParams:
    def test_the_basic_parameters_name_the_database_and_credentials(self):
        assert connection._postgres_connect_params("db", "h", 5432, "user", _FAKE_PASSWORD, None, None) == {
            "dbname": "db",
            "host": "h",
            "port": 5432,
            "user": "user",
            "password": _FAKE_PASSWORD,
        }

    def test_an_ssl_mode_and_a_schema_add_their_options(self):
        params = connection._postgres_connect_params("db", "h", 5432, "user", _FAKE_PASSWORD, "sch", "require")
        assert params["sslmode"] == "require"
        assert params["options"] == "--search_path=sch,public"


class TestCursorType:
    def test_a_plain_cursor_is_keyed_by_its_class_name(self):
        assert connection.get_cursor_type(None) == "cursor"

    def test_a_named_cursor_gets_its_own_key(self):
        assert connection.get_cursor_type(None, "named") == "cursor_named"


class TestDbFlavor:
    @pytest.mark.parametrize("flavor", ["postgres", "mysql"])
    def test_the_supported_flavours_are_accepted(self, flavor):
        assert connection._check_db_flavor(flavor) is None

    def test_any_other_flavour_is_refused(self):
        with pytest.raises(AssertionError, match="postgres"):
            connection._check_db_flavor("oracle")
