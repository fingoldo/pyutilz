"""Scanner tests for hardcoded_test_path, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_hardcoded_absolute_path_in_test,
)

from ._helpers import _write

# ---- hardcoded_absolute_path_in_test -------------------------------------


def test_hardcoded_absolute_path_windows_drive_letter_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "D:\\\\Machine Learning\\\\data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "hardcoded_absolute_path_in_test"
    assert findings[0].severity == "P2"


def test_hardcoded_absolute_path_posix_home_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/home/alice/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_macos_users_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/Users/bob/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_root_flagged(tmp_path: Path):
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/root/data.csv"
    assert p
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1


def test_hardcoded_absolute_path_ignores_non_test_file(tmp_path: Path):
    """The same literal in a non-test module is not flagged -- only test files are scanned."""
    _write(tmp_path, "helper.py", '''
def get_default_path():
    return "D:\\\\Machine Learning\\\\data.csv"
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_tmp_var_rooted_is_clean(tmp_path: Path):
    """A /tmp/-rooted or /var/-rooted literal is common/portable and NOT flagged."""
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture():
    p = "/tmp/scratch/data.csv"
    assert p
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_tmp_path_fixture_is_clean(tmp_path: Path):
    """The correct pattern (tmp_path/Path(__file__).parent-derived) is never flagged."""
    _write(tmp_path, "test_thing.py", '''
def test_uses_fixture(tmp_path):
    p = tmp_path / "data.csv"
    assert p
''')
    assert scan_hardcoded_absolute_path_in_test(tmp_path) == []


def test_hardcoded_absolute_path_detects_by_tests_directory(tmp_path: Path):
    """A file under a 'tests' directory is scanned even without a test_* / *_test.py name."""
    sub = tmp_path / "tests"
    sub.mkdir()
    _write(sub, "fixtures.py", '''
DATA_PATH = "C:/Users/carol/fixture.csv"
''')
    findings = scan_hardcoded_absolute_path_in_test(tmp_path)
    assert len(findings) == 1
