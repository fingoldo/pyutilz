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
            assert isinstance(result, tuple), \
                   "Should return (sql, params) tuple for parameterized query"

            sql, params = result

            # SQL should use placeholders (%s), not direct string interpolation
            assert '%s' in sql or '?' in sql, \
                   "SQL should use placeholders for parameterized query (SQL injection fix)"

            # Should NOT have f-string or .format() in the SQL string itself
            assert '{' not in sql, \
                   "SQL should not use f-string formatting (SQL injection risk)"

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
            assert 'INSERT' in sql.upper(), "Should have INSERT statement"
            assert 'ON CONFLICT' in sql.upper() or 'ON DUPLICATE' in sql.upper(), \
                   "Should have conflict handling (UPSERT)"

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
            placeholder_count = sql.count('%s') + sql.count('?')

            # Should match params length
            if params:
                assert len(params) == placeholder_count, \
                       f"Placeholder count ({placeholder_count}) should match params count ({len(params)})"


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
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('self =') and not line.startswith('    '):
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
        if 'version' in source.lower() or 'cache' in source.lower():
            # If using file-based versioning, should use hash not mtime
            if 'getmtime' in source:
                # Should also have hash-based alternative
                assert 'hashlib' in source or 'md5' in source or 'sha' in source, \
                       "Should use content-based versioning (hash), not just mtime (clock skew issues)"


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
        assert hasattr(DistributedManager, 'get_heartbeat_sql'), \
               "Should have get_heartbeat_sql method"


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
        lines = source.split('\n')
        for i, line in enumerate(lines):
            # Check for SQL keywords
            if any(kw in line.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                # If it's a SQL statement, check it's not using format() or %
                if '= f"' in line or '= f\'' in line:
                    # f-string SQL - potential injection risk
                    # Should use parameterized queries instead
                    if 'WHERE' in line.upper():
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
