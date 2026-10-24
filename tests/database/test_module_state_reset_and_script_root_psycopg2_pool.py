import importlib.util
import sys
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent


_spec = importlib.util.spec_from_file_location("pyutilz_tests_conftest_probe", _ROOT / "tests" / "conftest.py")


conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)


def test_reset_fixture_does_not_import_absent_modules(monkeypatch):
    gen = conftest._reset_process_lifetime_module_state.__wrapped__()
    next(gen)
    monkeypatch.delitem(sys.modules, "pyutilz.database.psycopg2_pool", raising=False)
    with pytest.raises(StopIteration):
        next(gen)
    assert "pyutilz.database.psycopg2_pool" not in sys.modules
