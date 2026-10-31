"""env_flag: one parser for boolean environment switches, and the call sites that now use it."""

import pytest

from pyutilz.core.env_flag import env_flag


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "On"])
def test_on_spellings_are_true(monkeypatch, raw):
    monkeypatch.setenv("PYUTILZ_TEST_FLAG", raw)
    assert env_flag("PYUTILZ_TEST_FLAG") is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off", " OFF "])
def test_off_spellings_are_false_even_when_default_is_on(monkeypatch, raw):
    monkeypatch.setenv("PYUTILZ_TEST_FLAG", raw)
    assert env_flag("PYUTILZ_TEST_FLAG", default=True) is False


@pytest.mark.parametrize("default", [True, False])
def test_unset_empty_or_unknown_returns_default(monkeypatch, default):
    monkeypatch.delenv("PYUTILZ_TEST_FLAG", raising=False)
    assert env_flag("PYUTILZ_TEST_FLAG", default=default) is default
    monkeypatch.setenv("PYUTILZ_TEST_FLAG", "")
    assert env_flag("PYUTILZ_TEST_FLAG", default=default) is default
    monkeypatch.setenv("PYUTILZ_TEST_FLAG", "maybe")
    assert env_flag("PYUTILZ_TEST_FLAG", default=default) is default


def test_zero_does_not_switch_the_gpu_opt_out_on(monkeypatch):
    from pyutilz.performance.kernel_tuning.cache import cache_base

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PYUTILZ_DISABLE_GPU", "0")
    assert cache_base._gpu_opted_out() is False
    monkeypatch.setenv("PYUTILZ_DISABLE_GPU", "1")
    assert cache_base._gpu_opted_out() is True


def test_write_prompt_closes_stdin_when_write_fails_unexpectedly():
    import io

    from pyutilz.llm import claude_code_cli

    class _Stdin(io.StringIO):
        def write(self, s):
            raise OSError("disk gone")

    class _Proc:
        stdin = _Stdin()

    with pytest.raises(OSError, match="disk gone"):
        claude_code_cli._write_prompt(_Proc(), "hello")
    assert _Proc.stdin.closed


def test_the_default_is_off_when_the_caller_omits_it(monkeypatch):
    """Callers that leave ``default`` out get False: a switch nobody set must stay off."""
    monkeypatch.delenv("PYUTILZ_TEST_FLAG", raising=False)
    assert env_flag("PYUTILZ_TEST_FLAG") is False
