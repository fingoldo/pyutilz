"""Behaviour of ``pyutilz.dev.notebook_init``: the polars thread budget is half the physical cores and says so when
polars is already imported, the magic runner prefers ``run_line_magic`` over the removed ``magic()``, the
extension loader degrades to a message outside IPython, and the package importer tolerates missing packages.
IPython is replaced by a recording fake, so no kernel is needed."""

from __future__ import annotations

import pytest

from pyutilz.dev import notebook_init as nb


class _FakeShell:
    def __init__(self, fail_on: str = ""):
        self.lines: list[str] = []
        self.fail_on = fail_on

    def run_line_magic(self, name: str, rest: str) -> None:
        line = f"{name} {rest}".strip()
        self.lines.append(line)
        if self.fail_on and line.startswith(self.fail_on):
            raise RuntimeError(f"{line} failed")


class _LegacyShell:
    """Mimics an IPython without ``run_line_magic``: only the removed-style ``magic()`` is available."""

    def __init__(self):
        self.lines: list[str] = []

    def magic(self, line: str) -> None:
        self.lines.append(line)


@pytest.fixture
def clean_env(monkeypatch):
    # setenv then delenv: the undo restores the original absent state, so values the code under test writes
    # into os.environ are removed again at teardown instead of leaking into later tests.
    for name in ("POLARS_MAX_THREADS", "_RJEM_MALLOC_CONF"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    return monkeypatch


class TestPolarsConfig:
    def test_the_thread_budget_is_half_the_physical_cores(self, clean_env, capsys):
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 8 if not logical else 16)
        clean_env.delitem(nb.sys.modules, "polars", raising=False)
        nb.setup_polars_config()
        assert nb.os.environ["POLARS_MAX_THREADS"] == "4"
        assert "Using 4 polars threads" in capsys.readouterr().out

    def test_a_single_core_machine_still_gets_one_thread(self, clean_env):
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 1)
        clean_env.delitem(nb.sys.modules, "polars", raising=False)
        nb.setup_polars_config()
        assert nb.os.environ["POLARS_MAX_THREADS"] == "1"

    def test_a_warning_is_printed_when_polars_is_already_imported(self, clean_env, capsys):
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 4)
        clean_env.setitem(nb.sys.modules, "polars", object())
        nb.setup_polars_config()
        assert "will NOT apply to this session" in capsys.readouterr().out

    def test_the_jemalloc_decay_setting_is_applied(self, clean_env):
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 2)
        nb.setup_polars_config()
        assert nb.os.environ["_RJEM_MALLOC_CONF"] == "muzzy_decay_ms:500"


class TestRunMagic:
    def test_run_line_magic_is_preferred_and_gets_the_name_and_argument_split(self):
        shell = _FakeShell()
        nb._run_magic(shell, "autoreload 2")
        assert shell.lines == ["autoreload 2"]

    def test_a_shell_without_run_line_magic_falls_back_to_magic(self):
        shell = _LegacyShell()
        nb._run_magic(shell, "load_ext autotime")
        assert shell.lines == ["load_ext autotime"]


class TestMinimalExtensions:
    def test_outside_ipython_the_loader_only_says_so(self, monkeypatch, capsys):
        monkeypatch.setattr(nb, "get_ipython", lambda: None)
        nb.load_jupyter_extensions_minimal()
        assert "Not running in IPython environment" in capsys.readouterr().out

    def test_autoreload_is_loaded_and_enabled(self, monkeypatch, capsys):
        shell = _FakeShell()
        monkeypatch.setattr(nb, "get_ipython", lambda: shell)
        nb.load_jupyter_extensions_minimal()
        assert shell.lines == ["load_ext autoreload", "autoreload 2"]
        assert "Loaded autoreload extension" in capsys.readouterr().out

    def test_a_failure_is_reported_not_raised(self, monkeypatch, capsys):
        shell = _FakeShell(fail_on="load_ext")
        monkeypatch.setattr(nb, "get_ipython", lambda: shell)
        nb.load_jupyter_extensions_minimal()
        assert "Failed to load autoreload" in capsys.readouterr().out


class TestFullExtensions:
    def test_a_failing_extension_does_not_skip_the_rest(self, monkeypatch, capsys):
        shell = _FakeShell(fail_on="load_ext line_profiler")
        monkeypatch.setattr(nb, "get_ipython", lambda: shell)
        nb.load_jupyter_extensions()
        assert "load_ext autotime" in shell.lines
        assert "autoreload 2" in shell.lines
        assert "Failed to load" in capsys.readouterr().out


class TestCommonPackages:
    def test_available_packages_are_returned_under_short_names(self):
        packages = nb.import_common_packages()
        assert "pd" in packages and "np" in packages

    def test_a_missing_package_is_skipped_and_reported(self, monkeypatch, capsys):
        monkeypatch.setitem(nb.sys.modules, "seaborn", None)
        packages = nb.import_common_packages()
        assert "sns" not in packages
        assert "seaborn not available" in capsys.readouterr().out


class TestInitNotebook:
    def test_without_imports_an_empty_mapping_is_returned(self, monkeypatch, clean_env):
        monkeypatch.setattr(nb, "get_ipython", lambda: None)
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 2)
        assert nb.init_notebook(include_imports=False) == {}

    def test_with_imports_the_packages_are_returned(self, monkeypatch, clean_env):
        monkeypatch.setattr(nb, "get_ipython", lambda: None)
        clean_env.setattr(nb.psutil, "cpu_count", lambda logical=True: 2)
        assert "pd" in nb.init_notebook(include_imports=True)
