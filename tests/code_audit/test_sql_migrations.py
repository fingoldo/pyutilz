"""Scanner tests for sql_migrations, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_sql_migration_idempotency,
)

from ._helpers import _write

# ---- sql_migration_not_idempotent ------------------------------------------


def test_migration_drop_constraint_without_if_exists_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets DROP CONSTRAINT widgets_pkey;\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert "DROP CONSTRAINT" in findings[0].detail


def test_migration_drop_constraint_if_exists_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", "ALTER TABLE widgets DROP CONSTRAINT IF EXISTS widgets_pkey;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_without_if_not_exists_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets ADD COLUMN status TEXT;\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "ADD COLUMN" in findings[0].detail


def test_migration_add_column_if_not_exists_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", "ALTER TABLE widgets ADD COLUMN IF NOT EXISTS status TEXT;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_primary_key_without_do_block_flags(tmp_path: Path):
    _write(tmp_path, "bad.sql", "ALTER TABLE widgets ADD PRIMARY KEY (id);\n")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "PRIMARY KEY" in findings[0].detail


def test_migration_add_primary_key_with_do_block_clean(tmp_path: Path):
    _write(tmp_path, "ok.sql", """
DO $$
BEGIN
    ALTER TABLE widgets ADD PRIMARY KEY (id);
END $$;
""")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_in_existence_guarded_do_block_clean(tmp_path: Path):
    """ADD COLUMN inside a DO $$ ... END $$ block that itself probes
    information_schema via IF NOT EXISTS is idempotent at the block level,
    even though the ALTER statement's own line has no IF NOT EXISTS
    keyword -- the classic existence-probe idiom."""
    _write(tmp_path, "ok.sql", """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'widgets' AND column_name = 'status'
    ) THEN
        ALTER TABLE widgets ADD COLUMN status TEXT;
    END IF;
END $$;
""")
    assert scan_sql_migration_idempotency(tmp_path) == []


def test_migration_add_column_in_plain_do_block_without_guard_flags(tmp_path: Path):
    """A DO block with no existence check at all does not make the ALTER
    inside it idempotent -- must still be flagged."""
    _write(tmp_path, "bad.sql", """
DO $$
BEGIN
    ALTER TABLE widgets ADD COLUMN status TEXT;
END $$;
""")
    findings = scan_sql_migration_idempotency(tmp_path)
    assert len(findings) == 1
    assert "ADD COLUMN" in findings[0].detail


def test_migration_non_sql_file_ignored(tmp_path: Path):
    _write(tmp_path, "notes.txt", "ALTER TABLE widgets DROP CONSTRAINT widgets_pkey;\n")
    assert scan_sql_migration_idempotency(tmp_path) == []


# ---- F30/F100: sql migrations -------------------------------------------------------


def test_sql_migration_idempotency_survives_an_excluded_ancestor_directory(tmp_path: Path):
    root = tmp_path / "build" / "proj"
    root.mkdir(parents=True)
    (root / "001.sql").write_text("ALTER TABLE t DROP COLUMN c;\n", encoding="utf-8")
    assert len(scan_sql_migration_idempotency(root)) == 1


def test_sql_migration_idempotency_ignores_a_commented_out_statement(tmp_path: Path):
    (tmp_path / "002.sql").write_text("-- DROP TABLE legacy_users;  (removed in v2)\nSELECT 1;\n", encoding="utf-8")
    assert scan_sql_migration_idempotency(tmp_path) == []
