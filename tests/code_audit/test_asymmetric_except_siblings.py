"""Scanner tests for asymmetric_except_siblings, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.asymmetric_except_siblings import scan_asymmetric_except_siblings

from ._helpers import _write

# ---- asymmetric_except_siblings ------------------------------------------
#
# Run against a real repository this found `SafeDB.rollback` calling `_reconnect()` bare while
# `_retry_resource_error` wrapped the identical call -- and 21 of that package's 51 rollback call
# sites are inside an `except`, so a failing reconnect aborted whatever was recovering.


def test_asymmetric_except_siblings_flags_the_unguarded_twin(tmp_path: Path):
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def already_in_db(self, cid):
        try:
            return self.db.query(cid)
        except psycopg2.Error:
            self.db.rollback()
            return False

    def already_in_db_batch(self, cids):
        try:
            return self.db.query_many(cids)
        except psycopg2.Error:
            try:
                self.db.rollback()
            except Exception:
                pass
            return []
""",
    )
    findings = scan_asymmetric_except_siblings(tmp_path)
    assert len(findings) == 1
    assert "already_in_db" in findings[0].detail
    assert "rollback" in findings[0].detail


def test_asymmetric_except_siblings_needs_the_same_exception_type(tmp_path: Path):
    """Two handlers doing genuinely different jobs share neither the type nor the call, and
    comparing them would report every class with two try blocks in it."""
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except psycopg2.Error:
            self.db.rollback()

    def b(self):
        try:
            go()
        except OSError:
            try:
                self.db.rollback()
            except Exception:
                pass
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []


def test_asymmetric_except_siblings_is_silent_when_both_guard(tmp_path: Path):
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass

    def b(self):
        try:
            go()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                pass
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []


def test_asymmetric_except_siblings_needs_two_siblings(tmp_path: Path):
    """A lone unguarded handler is a judgement call, not drift. The rule reports only where the
    same class already does it the other way."""
    _write(
        tmp_path,
        "batch.py",
        """
class Scanner:
    def a(self):
        try:
            go()
        except Exception:
            self.db.rollback()
""",
    )
    assert scan_asymmetric_except_siblings(tmp_path) == []
