"""Reload-debounce, logging and cast-fallback pins for system.config (mutation survivors)."""

from __future__ import annotations

import logging
import os

import pytest

from pyutilz.system import config as cfgmod

_LOG = logging.getLogger("cfgtest")


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock(0.0)
    monkeypatch.setattr(cfgmod.time, "monotonic", fake)
    return fake


@pytest.fixture
def toml_path(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('[db]\nretries = 3\nname = "alpha"\n', encoding="utf-8")
    return path


def _bump(path, text):
    path.write_text(text, encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime + 10, stat.st_mtime + 10))


class TestDefaults:
    def test_the_default_check_interval_is_five_seconds(self, toml_path):
        assert cfgmod.TomlLiveConfig(toml_path).__dict__["_check_interval"] == 5.0

    def test_a_check_just_past_the_interval_from_the_start_reloads(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 9\nname = "alpha"\n')
        clock.now = 5.05
        assert cfg.get("db", "retries") == 9

    def test_a_check_before_the_interval_does_not_reload(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 9\nname = "alpha"\n')
        clock.now = 4.9
        assert cfg.get("db", "retries") == 3


class TestDebounceBoundaries:
    def test_a_check_exactly_at_the_interval_reloads(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, check_interval=5.0, logger=_LOG)
        clock.now = 10.0
        cfg.get("db", "retries")
        _bump(toml_path, '[db]\nretries = 7\nname = "alpha"\n')
        clock.now = 15.0
        assert cfg.get("db", "retries") == 7

    def test_a_check_inside_the_interval_is_skipped_even_with_a_large_stamp(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, check_interval=5.0, logger=_LOG)
        clock.now = 10.0
        cfg.get("db", "retries")
        _bump(toml_path, '[db]\nretries = 7\nname = "alpha"\n')
        clock.now = 12.0
        assert cfg.get("db", "retries") == 3


class TestMissingFile:
    def test_a_file_that_vanishes_keeps_the_previous_values_and_warns(self, toml_path, clock, caplog):
        caplog.set_level(logging.WARNING)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        toml_path.unlink()
        clock.now = 10.0
        assert cfg.get("db", "retries") == 3
        assert f"Config file {toml_path} unstattable" in caplog.text and "keeping previous values" in caplog.text

    def test_a_first_load_with_no_file_does_not_warn(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        cfgmod.TomlLiveConfig(tmp_path / "absent.toml", logger=_LOG)
        assert "not found" not in caplog.text

    def test_reload_of_a_vanished_file_keeps_values_and_names_the_path(self, toml_path, caplog):
        caplog.set_level(logging.WARNING)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        toml_path.unlink()
        cfg._reload()
        assert cfg._data["db"]["retries"] == 3
        assert f"Config file {toml_path} not found -- keeping previous values" in caplog.text


class TestUnreadable:
    def test_an_unreadable_path_keeps_previous_values_and_names_the_path(self, tmp_path, toml_path, clock, caplog):
        caplog.set_level(logging.WARNING)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        directory = tmp_path / "as_dir.toml"
        directory.mkdir()
        cfg._path = directory
        clock.now = 10.0
        assert cfg.get("db", "retries") == 3
        assert f"Config file {directory} unreadable (keeping previous values)" in caplog.text


class TestParseErrors:
    def test_a_parse_error_keeps_the_values_and_records_the_file_stamp(self, toml_path, caplog):
        caplog.set_level(logging.WARNING)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        toml_path.write_text("[db\nretries =", encoding="utf-8")
        cfg._reload()
        assert cfg._data["db"]["retries"] == 3
        assert "Config parse error (keeping previous values)" in caplog.text
        assert cfg._mtime == toml_path.stat().st_mtime


class TestReloadLogging:
    def test_a_changed_file_logs_the_reload_and_each_changed_key(self, toml_path, clock, caplog):
        caplog.set_level(logging.INFO)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 9\nname = "alpha"\n')
        clock.now = 10.0
        cfg.get("db", "retries")
        assert "Config file changed, reloading" in caplog.text
        assert "[config] [db].retries changed: 3 -> 9" in caplog.text

    def test_a_new_section_is_logged_as_a_section_change(self, toml_path, clock, caplog):
        caplog.set_level(logging.INFO)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 3\nname = "alpha"\n[extra]\nflag = 1\n')
        clock.now = 10.0
        cfg.get("db", "retries")
        assert "[config] [extra].flag changed: None -> 1" in caplog.text

    def test_a_top_level_scalar_is_logged_against_an_empty_table(self, toml_path, clock, caplog):
        caplog.set_level(logging.INFO)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, 'aaa = 1\n[db]\nretries = 3\nname = "alpha"\n')
        clock.now = 10.0
        cfg.get("db", "retries")
        assert "[config] [aaa] changed: {} -> 1" in caplog.text


class TestCastFallbacks:
    def test_a_bool_under_an_int_request_is_cast_not_aliased(self, tmp_path):
        path = tmp_path / "b.toml"
        path.write_text("[s]\nflag = true\n", encoding="utf-8")
        cfg = cfgmod.TomlLiveConfig(path, logger=_LOG)
        assert cfg.get("s", "flag", type_=int) == 1

    def test_a_lossy_float_to_int_cast_warns_with_the_truncated_value(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        path = tmp_path / "f.toml"
        path.write_text("[s]\ntimeout = 2.5\n", encoding="utf-8")
        cfg = cfgmod.TomlLiveConfig(path, logger=_LOG)
        assert cfg.get("s", "timeout", type_=int) == 2
        assert "float 2.5 truncated to int 2 by type_=int" in caplog.text
        assert "config [s].timeout" in caplog.text

    def test_a_bad_value_falls_back_to_the_caller_default_then_the_defaults(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        path = tmp_path / "bad.toml"
        path.write_text('[s]\nretries = "many"\n', encoding="utf-8")
        cfg = cfgmod.TomlLiveConfig(path, defaults={"s": {"retries": 4}}, logger=_LOG)
        assert cfg.get("s", "retries", default=None) == 4
        assert "config [s].retries: bad value 'many'" in caplog.text
        assert "using 4" in caplog.text
        assert cfg.get("s", "retries", default=7) == 7


class TestSectionAndDataReload:
    def test_get_section_reloads_a_changed_file(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 8\nname = "alpha"\n')
        clock.now = 10.0
        assert cfg.get_section("db")["retries"] == 8

    def test_data_reloads_a_changed_file(self, toml_path, clock):
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, '[db]\nretries = 8\nname = "alpha"\n')
        clock.now = 10.0
        assert cfg.data["db"]["retries"] == 8


class TestChangeLoggingContinues:
    def test_a_non_table_section_does_not_stop_the_logging_of_later_sections(self, toml_path, clock, caplog):
        caplog.set_level(logging.INFO)
        cfg = cfgmod.TomlLiveConfig(toml_path, logger=_LOG)
        _bump(toml_path, 'aaa = 1\n[db]\nretries = 9\nname = "alpha"\n')
        clock.now = 10.0
        cfg.get("db", "retries")
        assert "[config] [aaa] changed: {} -> 1" in caplog.text
        assert "[config] [db].retries changed: 3 -> 9" in caplog.text
