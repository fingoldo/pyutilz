import importlib.util
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent


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
