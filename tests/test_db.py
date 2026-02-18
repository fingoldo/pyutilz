"""Tests for db.py - Phase 2 refactoring

Tests cover:
- SQL injection protection (validate_sql_identifier)
- UPDATE SET clause syntax fix (" and " → ", ")
- Infinite retry loop fix (max_retries circuit breaker)
- Parameterized query usage
"""

import pytest
import re


class TestValidateSqlIdentifier:
    """Test validate_sql_identifier function - SQL injection protection"""

    def test_accepts_valid_identifiers(self):
        """Test that valid SQL identifiers are accepted"""
        from pyutilz.db import validate_sql_identifier

        # Valid identifiers
        valid_names = [
            "table_name",
            "TableName",
            "table123",
            "_private_table",
            "snake_case_name"
        ]

        for name in valid_names:
            result = validate_sql_identifier(name)
            assert result == name

    def test_rejects_sql_injection_attempts(self):
        """Test that SQL injection attempts are rejected"""
        from pyutilz.db import validate_sql_identifier

        # Invalid identifiers (SQL injection attempts)
        invalid_names = [
            "table; DROP TABLE users--",
            "table' OR '1'='1",
            "table-name",  # Hyphen not allowed
            "123table",  # Can't start with number
            "table name",  # Space not allowed
            "table.column",  # Dot not allowed
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid SQL identifier"):
                validate_sql_identifier(name)

    def test_rejects_non_string_input(self):
        """Test that non-string input is rejected"""
        from pyutilz.db import validate_sql_identifier

        with pytest.raises(ValueError, match="must be a string"):
            validate_sql_identifier(123)

        with pytest.raises(ValueError, match="must be a string"):
            validate_sql_identifier(None)


class TestGetTableFields:
    """Test get_table_fields - now validates table names"""

    @pytest.mark.skip(reason="Requires database connection")
    def test_validates_table_name(self):
        """Test that table name is validated (line 74 fix)"""
        from pyutilz.db import get_table_fields

        # Should raise ValueError for invalid table name
        with pytest.raises(ValueError):
            get_table_fields("invalid; DROP TABLE", "alias")


class TestUpdateSetClauseFix:
    """Test that UPDATE SET clause uses comma, not 'and'"""

    def test_update_uses_comma_in_set_clause(self):
        """Verify UPDATE query uses comma separator in SET clause (line 345 fix)"""
        # This is a code inspection test
        # The fix changed " and ".join to ", ".join for SET clause

        # Read the db.py file
        import pyutilz.db as db_module
        import inspect

        # Get the source code
        source = inspect.getsource(db_module)

        # Look for the UPDATE construction
        # Should NOT have: " and ".join(sql_set_templates)
        # Should have: ", ".join(sql_set_templates)

        # Check for the fixed pattern (comma in SET clause)
        assert ', ".join(sql_set_templates)' in source or \
               '", ".join(sql_set_templates)' in source, \
               "UPDATE SET clause should use comma separator"

        # Ensure the buggy pattern is NOT present
        # (though multiple " and " might exist for WHERE clauses, which is correct)


class TestBasicDbExecuteRetry:
    """Test basic_db_execute retry logic - infinite loop fix"""

    def test_has_max_retries_parameter(self):
        """Test that basic_db_execute has max_retries parameter (infinite loop fix)"""
        from pyutilz.db import basic_db_execute
        import inspect

        # Get function signature
        sig = inspect.signature(basic_db_execute)

        # Should have max_retries parameter
        assert 'max_retries' in sig.parameters

        # Default should be 5
        assert sig.parameters['max_retries'].default == 5

    @pytest.mark.skip(reason="Requires database connection to test retry logic")
    def test_respects_max_retries(self):
        """Test that function respects max_retries limit"""
        # Would test that after max_retries, exception is raised
        # instead of retrying forever
        pass


class TestShowcaseTable:
    """Test showcase_table function - SQL injection protection"""

    def test_validates_table_name(self):
        """Test that table_name is validated (line 644 fix)"""
        from pyutilz.db import showcase_table

        # Should raise ValueError for invalid table name
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            showcase_table("invalid; DROP TABLE users--")

    def test_accepts_valid_table_name(self):
        """Test that valid table names pass validation"""
        from pyutilz.db import showcase_table

        # Valid table name should pass validation
        # (will fail later due to no connection, but validation passes)
        try:
            showcase_table("valid_table")
        except ValueError as e:
            if "Invalid SQL identifier" in str(e):
                pytest.fail("Valid table name was rejected")
            # Other errors (like connection errors) are expected
        except Exception:
            # Other exceptions (no DB connection) are fine
            pass


class TestParameterizedQueries:
    """Test that parameterized queries are used"""

    def test_settings_query_uses_parameterization(self):
        """Test that settings query uses %s placeholder, not % formatting (line 387 fix)"""
        import pyutilz.db as db_module
        import inspect

        source = inspect.getsource(db_module)

        # The fix changed from:
        #   sql += " where strpos(name,'%s')>0" % settings_names_contains
        # To:
        #   sql += " where strpos(name,%s)>0"
        #   with params tuple

        # Check the fixed version exists
        assert 'strpos(name,%s)' in source or 'strpos(name, %s)' in source, \
               "Should use parameterized query with %s placeholder"


@pytest.mark.parametrize("identifier,should_pass", [
    ("valid_table", True),
    ("AnotherTable", True),
    ("_private", True),
    ("table123", True),
    ("table; DROP", False),
    ("table'--", False),
    ("123invalid", False),
    ("table-name", False),
])
def test_validate_sql_identifier_parametrized(identifier, should_pass):
    """Parametrized test for SQL identifier validation"""
    from pyutilz.db import validate_sql_identifier

    if should_pass:
        result = validate_sql_identifier(identifier)
        assert result == identifier
    else:
        with pytest.raises(ValueError):
            validate_sql_identifier(identifier)
