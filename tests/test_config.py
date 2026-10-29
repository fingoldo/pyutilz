"""Behaviour of ``pyutilz.system.config.TomlLiveConfig``: values are read from a TOML file with the caller's default and
the constructor's defaults as fallbacks, a value that cannot be cast falls back with a warning, a float truncated to int
is reported, booleans are not silently cast to int, lists and dicts come back as they are, and a hot-reload picks up a
changed file while a missing or unparsable file keeps the previous values. Every test uses a file under ``tmp_path``
and ``check_interval=0`` so each read checks the file's mtime."""

from __future__ import annotations

import logging
import os
import pickle

import pytest

from pyutilz.system.config import TomlLiveConfig


def _write(path, text, mtime):
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def toml_path(tmp_path):
    path = tmp_path / "config.toml"
    _write(path, '[net]\ntimeout = 30\nratio = 0.5\nenabled = true\nhosts = ["a", "b"]\nlimits = {max = 3}\n', 1_000_000)
    return path


@pytest.fixture
def cfg(toml_path):
    return TomlLiveConfig(toml_path, check_interval=0)


class TestGet:
    def test_a_value_from_the_file_is_returned_cast_to_the_requested_type(self, cfg):
        assert cfg.get("net", "timeout", type_=int) == 30

    def test_a_missing_key_falls_back_to_the_caller_default(self, cfg):
        assert cfg.get("net", "absent", default=7) == 7

    def test_a_missing_key_falls_back_to_the_constructor_defaults(self, toml_path):
        cfg = TomlLiveConfig(toml_path, check_interval=0, defaults={"net": {"absent": 9}})
        assert cfg.get("net", "absent") == 9

    def test_a_missing_key_with_no_fallback_is_none(self, cfg):
        assert cfg.get("net", "absent") is None

    def test_a_missing_section_uses_the_defaults_for_its_key(self, toml_path):
        cfg = TomlLiveConfig(toml_path, check_interval=0, defaults={"other": {"k": "v"}})
        assert cfg.get("other", "k") == "v"

    def test_lists_and_dicts_are_returned_as_they_are(self, cfg):
        assert cfg.get("net", "hosts") == ["a", "b"]
        assert cfg.get("net", "limits") == {"max": 3}

    def test_a_float_read_as_int_is_truncated_and_reported(self, cfg, caplog):
        with caplog.at_level(logging.WARNING):
            assert cfg.get("net", "ratio", type_=int) == 0
        assert any("truncated to int" in rec.getMessage() for rec in caplog.records)

    def test_a_float_read_as_float_keeps_its_precision(self, cfg):
        assert cfg.get("net", "ratio", type_=float) == 0.5

    def test_a_bool_is_not_silently_aliased_to_an_int(self, cfg):
        value = cfg.get("net", "enabled", type_=int)
        assert value == 1 and type(value) is int

    def test_an_uncastable_string_uses_the_caller_default(self, toml_path, caplog):
        _write(toml_path, '[net]\nname = "abc"\n', 2_000_000)
        cfg = TomlLiveConfig(toml_path, check_interval=0)
        with caplog.at_level(logging.WARNING):
            assert cfg.get("net", "name", default=3, type_=int) == 3
        assert any("bad value" in rec.getMessage() for rec in caplog.records)


class TestGetSection:
    def test_a_section_is_returned_as_a_copy(self, cfg):
        section = cfg.get_section("net")
        section["timeout"] = 999
        assert cfg.get("net", "timeout") == 30

    def test_a_missing_section_gives_the_defaults_for_it(self, toml_path):
        cfg = TomlLiveConfig(toml_path, check_interval=0, defaults={"extra": {"k": 1}})
        assert cfg.get_section("extra") == {"k": 1}
        assert cfg.get_section("nothing") == {}


class TestDataProperty:
    def test_data_is_a_shallow_copy_of_the_whole_config(self, cfg):
        snapshot = cfg.data
        snapshot["net"] = {}
        assert cfg.get("net", "timeout") == 30

    def test_the_setter_stores_a_copy_of_the_value(self, cfg):
        new = {"net": {"timeout": 45}}
        cfg.data = new
        new["net"] = {}
        assert cfg.get("net", "timeout") == 45


class TestHotReload:
    def test_a_changed_file_is_reloaded_on_the_next_read(self, cfg, toml_path):
        _write(toml_path, "[net]\ntimeout = 60\n", 3_000_000)
        assert cfg.get("net", "timeout") == 60

    def test_a_file_that_becomes_unparsable_keeps_the_previous_values(self, cfg, toml_path, caplog):
        _write(toml_path, "[net\ntimeout = ", 4_000_000)
        with caplog.at_level(logging.WARNING):
            assert cfg.get("net", "timeout") == 30
        assert any("parse error" in rec.getMessage() for rec in caplog.records)

    def test_a_deleted_file_keeps_the_previous_values_with_a_warning(self, cfg, toml_path, caplog):
        toml_path.unlink()
        with caplog.at_level(logging.WARNING):
            assert cfg.get("net", "timeout") == 30
        assert any("keeping previous values" in rec.getMessage() for rec in caplog.records)

    def test_a_file_that_is_missing_at_start_gives_only_the_defaults(self, tmp_path):
        cfg = TomlLiveConfig(tmp_path / "absent.toml", check_interval=0, defaults={"net": {"timeout": 11}})
        assert cfg.get("net", "timeout") == 11

    def test_a_change_is_logged_with_its_old_and_new_values(self, cfg, toml_path, caplog):
        _write(toml_path, "[net]\ntimeout = 60\n", 5_000_000)
        with caplog.at_level(logging.INFO):
            cfg.get("net", "timeout")
        assert any("[config] [net].timeout changed: 30 -> 60" in rec.getMessage() for rec in caplog.records)


class TestPickling:
    def test_a_config_survives_a_pickle_round_trip_with_a_working_lock(self, cfg):
        restored = pickle.loads(pickle.dumps(cfg))
        assert restored.get("net", "timeout") == 30
