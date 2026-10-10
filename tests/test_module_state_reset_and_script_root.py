import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pyutilz_tests_conftest_probe", _ROOT / "tests" / "conftest.py")
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)


def test_reset_fixture_clears_registered_module_state():
    from pyutilz.dev import logginglib
    from pyutilz.llm import _claude_models

    gen = conftest._reset_process_lifetime_module_state.__wrapped__()
    next(gen)
    logginglib._log_throttle_last["k"] = 1.0
    _claude_models._warned_unknown.add("m")
    with pytest.raises(StopIteration):
        next(gen)
    assert not logginglib._log_throttle_last
    assert not _claude_models._warned_unknown


def test_reset_fixture_does_not_import_absent_modules(monkeypatch):
    gen = conftest._reset_process_lifetime_module_state.__wrapped__()
    next(gen)
    monkeypatch.delitem(sys.modules, "pyutilz.database.psycopg2_pool", raising=False)
    with pytest.raises(StopIteration):
        next(gen)
    assert "pyutilz.database.psycopg2_pool" not in sys.modules


def test_prove_meta_checks_repo_is_derived_from_file():
    spec = importlib.util.spec_from_file_location("prove_meta_checks_probe", _ROOT / "scripts" / "prove_meta_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.REPO == _ROOT
