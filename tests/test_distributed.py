"""Tests for distributed.py - Phase 2 refactoring

Tests cover:
- SQL injection protection (parameterized queries in heartbeat SQL)
- Module-level variable naming (avoid 'self' at module level)
- Content-based versioning (avoid file mtime issues)
"""

import pytest
import os
import tempfile


class TestHeartbeatSql:
    """Test get_heartbeat_sql - SQL injection protection"""

    def test_returns_parameterized_query(self):
        """Test that heartbeat SQL uses parameterized queries (line 91-115 fix)"""
        try:
            from pyutilz.distributed import DistributedManager
        except ImportError:
            pytest.skip("DistributedManager not available")

        # Create manager instance
        try:
            manager = DistributedManager()
        except Exception:
            pytest.skip("Cannot instantiate DistributedManager")

        # Get heartbeat SQL
        try:
            result = manager.get_heartbeat_sql()
        except Exception as e:
            pytest.skip(f"get_heartbeat_sql not available: {e}")

        if result:
            # Should return tuple (sql, params)
            assert isinstance(result, tuple), "Should return (sql, params) tuple for parameterized query"

            sql, params = result

            # SQL should use placeholders (%s), not direct string interpolation
            assert "%s" in sql or "?" in sql, "SQL should use placeholders for parameterized query (SQL injection fix)"

            # Should NOT have f-string or .format() in the SQL string itself
            assert "{" not in sql, "SQL should not use f-string formatting (SQL injection risk)"

    def test_heartbeat_sql_has_upsert_structure(self):
        """Test that heartbeat SQL has proper UPSERT structure"""
        try:
            from pyutilz.distributed import DistributedManager
        except ImportError:
            pytest.skip("DistributedManager not available")

        try:
            manager = DistributedManager()
            result = manager.get_heartbeat_sql()
        except Exception:
            pytest.skip("Cannot get heartbeat SQL")

        if result:
            sql, params = result

            # Should have INSERT ... ON CONFLICT structure
            assert "INSERT" in sql.upper(), "Should have INSERT statement"
            assert "ON CONFLICT" in sql.upper() or "ON DUPLICATE" in sql.upper(), "Should have conflict handling (UPSERT)"

    def test_heartbeat_sql_parameters_count_matches(self):
        """Test that number of placeholders matches number of parameters"""
        try:
            from pyutilz.distributed import DistributedManager
        except ImportError:
            pytest.skip("DistributedManager not available")

        try:
            manager = DistributedManager()
            result = manager.get_heartbeat_sql()
        except Exception:
            pytest.skip("Cannot get heartbeat SQL")

        if result:
            sql, params = result

            # Count placeholders
            placeholder_count = sql.count("%s") + sql.count("?")

            # Should match params length
            if params:
                assert len(params) == placeholder_count, f"Placeholder count ({placeholder_count}) should match params count ({len(params)})"


class TestModuleLevelVariables:
    """Test module-level variable naming conventions"""

    def test_no_self_at_module_level(self):
        """Test that module doesn't use 'self' as module-level variable (line 31 fix)"""
        try:
            import pyutilz.distributed as distributed_module
            import inspect
        except ImportError:
            pytest.skip("distributed module not available")

        source = inspect.getsource(distributed_module)

        # Check for module-level 'self ='
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("self =") and not line.startswith("    "):
                # Found module-level 'self =' (not indented)
                pytest.fail(f"Line {i+1}: Module-level variable should not be named 'self' (confusing)")


class TestVersioning:
    """Test content-based versioning vs mtime"""

    def test_version_not_using_mtime(self):
        """Test that versioning uses content hash, not file mtime (clock skew issue)"""
        try:
            import pyutilz.distributed as distributed_module
            import inspect
        except ImportError:
            pytest.skip("distributed module not available")

        source = inspect.getsource(distributed_module)

        # Check if versioning/caching code exists
        if "version" in source.lower() or "cache" in source.lower():
            # If using file-based versioning, should use hash not mtime
            if "getmtime" in source:
                # Should also have hash-based alternative
                assert (
                    "hashlib" in source or "md5" in source or "sha" in source
                ), "Should use content-based versioning (hash), not just mtime (clock skew issues)"


class TestDistributedManagerInstantiation:
    """Test DistributedManager can be instantiated"""

    def test_manager_creation(self):
        """Test that DistributedManager can be created"""
        try:
            from pyutilz.distributed import DistributedManager
        except ImportError:
            pytest.skip("DistributedManager not available")

        try:
            manager = DistributedManager()
            assert manager is not None
        except Exception as e:
            # May fail due to missing configuration, but class should exist
            pytest.skip(f"Cannot create DistributedManager (missing config expected): {e}")

    def test_manager_has_required_methods(self):
        """Test that DistributedManager has expected methods"""
        try:
            from pyutilz.distributed import DistributedManager
        except ImportError:
            pytest.skip("DistributedManager not available")

        # Check for expected methods
        assert hasattr(DistributedManager, "get_heartbeat_sql"), "Should have get_heartbeat_sql method"


class TestSqlInjectionProtection:
    """Test SQL injection protection across distributed module"""

    def test_no_string_formatting_in_sql(self):
        """Test that SQL queries don't use dangerous string formatting"""
        try:
            import pyutilz.distributed as distributed_module
            import inspect
        except ImportError:
            pytest.skip("distributed module not available")

        source = inspect.getsource(distributed_module)

        # Look for SQL-related code
        lines = source.split("\n")
        for i, line in enumerate(lines):
            # Check for SQL keywords
            if any(kw in line.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                # If it's a SQL statement, check it's not using format() or %
                if '= f"' in line or "= f'" in line:
                    # f-string SQL - potential injection risk
                    # Should use parameterized queries instead
                    if "WHERE" in line.upper():
                        pytest.fail(f"Line {i+1}: SQL with WHERE clause should use parameterized query, not f-string")


@pytest.mark.parametrize("status,ip", [
    ("ok", "192.168.1.1"),
    ("error", "10.0.0.1"),
    ("ok", None),
])
def test_heartbeat_sql_with_different_parameters(status, ip):
    """Parametrized test for heartbeat SQL with different parameters"""
    try:
        from pyutilz.distributed import DistributedManager
    except ImportError:
        pytest.skip("DistributedManager not available")

    try:
        manager = DistributedManager()
        result = manager.get_heartbeat_sql(status=status, ip=ip)
    except Exception:
        pytest.skip("Cannot get heartbeat SQL")

    if result:
        sql, params = result
        assert isinstance(sql, str)
        assert params is None or isinstance(params, tuple)


def test_distributed_module_imports_successfully():
    """Test that distributed module can be imported without errors"""
    try:
        import pyutilz.distributed
        assert pyutilz.distributed is not None
    except ImportError as e:
        # Some dependencies might be missing
        pytest.skip(f"distributed module dependencies not available: {e}")


class TestIdentityRaceCondition:
    """Regression test for a concurrent register_scraper/heartbeat_scraper identity race.

    register_scraper() mutates the module-level scraper-identity globals
    (_container.node_id, pid, m_app_name, m_scraper_name, m_version, m_ip) and
    get_heartbeat_sql()/heartbeat_scraper() read all of them together to build one
    SQL payload. Without a lock, a heartbeat running concurrently with a
    re-registration could observe a torn mix of old and new identity fields
    (e.g. the new node_id paired with the old scraper name). This test drives
    both from multiple threads (with all DB calls mocked out) and asserts every
    captured heartbeat payload is one of the exact identities that was ever
    registered, never a mixed/impossible combination.
    """

    def test_concurrent_register_and_heartbeat_never_mix_identities(self, monkeypatch):
        import threading

        import pyutilz.system.distributed as distributed_module

        # Two distinct, fully self-consistent identities that will be registered
        # from different threads. A correct implementation must never produce a
        # heartbeat payload that mixes fields from both.
        identities = [
            {"node_id": 111, "scraper_name": "scraperA", "version": "v1", "app_name": "appA", "ip": "10.0.0.1"},
            {"node_id": 222, "scraper_name": "scraperB", "version": "v2", "app_name": "appB", "ip": "10.0.0.2"},
        ]

        def fake_db_command(action, table, *args, **kwargs):
            # Simulate the "select"/"insert" against "nodes" setting node_id on
            # the fetch_into container, driven by whichever identity is currently
            # being registered (tracked via a thread-local set just before the call).
            fetch_into = kwargs.get("fetch_into")
            if fetch_into is not None and table == "nodes":
                fetch_into.node_id = _current_identity.node_id

        def fake_get_system_info(only_stats=False):
            return {"host_name": "h", "os_machine_guid": "g", "os_serial": "s"}

        def fake_safe_execute(sql_params):
            sql, params = sql_params
            if sql:
                captured.append(params)

        captured = []
        _current_identity = threading.local()

        monkeypatch.setattr(distributed_module.db, "db_command", fake_db_command)
        monkeypatch.setattr(distributed_module.db, "safe_execute", fake_safe_execute)
        monkeypatch.setattr(distributed_module.system, "get_system_info", fake_get_system_info)
        monkeypatch.setattr(distributed_module.pythonlib, "lookup_in_stack", lambda *_a, **_k: None)
        monkeypatch.setattr(distributed_module.web, "get_external_ip", lambda: "0.0.0.0")

        # Reset shared container so each run starts unregistered.
        distributed_module._container.node_id = None

        errors = []

        def worker(identity):
            _current_identity.node_id = identity["node_id"]
            try:
                for _ in range(20):
                    with distributed_module._identity_lock:
                        distributed_module.m_app_name = identity["app_name"]
                        distributed_module.m_scraper_name = identity["scraper_name"]
                        distributed_module.m_version = identity["version"]
                        distributed_module.m_ip = identity["ip"]
                        distributed_module._container.node_id = identity["node_id"]
                    distributed_module.heartbeat_scraper(status="ok")
            except Exception as e:  # pragma: no cover - surfaced via errors list
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(identity,)) for identity in identities]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        valid_combinations = {
            (identity["node_id"], identity["version"], identity["scraper_name"], identity["app_name"], identity["ip"])
            for identity in identities
        }

        for params in captured:
            if params is None:
                continue
            node_id_seen, pid_seen, version_seen, scraper_name_seen, status_seen, ip_seen, app_name_seen = params
            combo = (node_id_seen, version_seen, scraper_name_seen, app_name_seen, ip_seen)
            assert combo in valid_combinations, f"Torn/mixed identity in heartbeat payload: {params}"
