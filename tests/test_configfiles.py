"""Round-trip and read behaviour of ``pyutilz.text.strings.configfiles``: base64 encoding by default,
literal coercion on read, section prefixing, append versus replace on write, and the failure returns."""

from __future__ import annotations

import base64
import logging

import pytest

from pyutilz.text.strings.configfiles import read_config_file, write_config_file


@pytest.fixture
def ini_path(tmp_path):
    return str(tmp_path / "settings.ini")


class TestWriteThenRead:
    def test_base64_round_trip_restores_values_and_literal_types(self, ini_path):
        values = {"count": 3, "ratio": 0.5, "flag": True, "names": ["a", "b"]}
        assert write_config_file(ini_path, values, section="MAIN") is True
        out: dict = {}
        assert read_config_file(ini_path, out, section="MAIN") is True
        assert out == values

    def test_values_are_stored_base64_encoded_by_default(self, ini_path):
        write_config_file(ini_path, {"token": "hello"}, section="MAIN")
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out["token"] == base64.b64encode(b"hello").decode()

    def test_plain_mode_stores_the_value_verbatim(self, ini_path):
        write_config_file(ini_path, {"greeting": "hello"}, section="MAIN", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out["greeting"] == "hello"

    def test_percent_signs_survive_a_round_trip_unescaped(self, ini_path):
        write_config_file(ini_path, {"discount": "50% off"}, section="MAIN", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out["discount"] == "50% off"


class TestRead:
    def test_a_named_section_keeps_plain_keys(self, ini_path):
        write_config_file(ini_path, {"host": "db"}, section="DEV", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="DEV", encryption=None)
        assert out == {"host": "db"}

    def test_all_sections_are_prefixed_by_their_lowercased_name(self, ini_path):
        write_config_file(ini_path, {"host": "dev-db"}, section="DEV", encryption=None)
        write_config_file(ini_path, {"host": "prod-db"}, section="PROD", encryption=None)
        out: dict = {}
        assert read_config_file(ini_path, out, section=None, encryption=None) is True
        assert out == {"dev_host": "dev-db", "prod_host": "prod-db"}

    def test_variables_as_a_comma_string_select_only_those_keys(self, ini_path):
        write_config_file(ini_path, {"a": 1, "b": 2, "c": 3}, section="MAIN", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", variables="a,c", encryption=None)
        assert out == {"a": 1, "c": 3}

    def test_a_missing_variable_is_stored_as_none_under_its_section_key(self, ini_path):
        write_config_file(ini_path, {"a": 1}, section="MAIN", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", variables="a,missing", encryption=None)
        assert out == {"a": 1, "missing": None}

    def test_a_section_argument_of_the_wrong_type_returns_none(self, ini_path):
        write_config_file(ini_path, {"a": 1}, section="MAIN", encryption=None)
        assert read_config_file(ini_path, {}, section=5) is None

    def test_a_missing_file_reads_nothing_and_still_succeeds(self, tmp_path):
        out: dict = {}
        assert read_config_file(str(tmp_path / "absent.ini"), out, section=None) is True
        assert out == {}


class TestWriteModes:
    def test_append_merges_into_an_existing_section(self, ini_path):
        write_config_file(ini_path, {"a": 1}, section="MAIN", encryption=None)
        write_config_file(ini_path, {"b": 2}, section="MAIN", encryption=None, mode="append")
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out == {"a": 1, "b": 2}

    def test_replace_discards_the_previous_contents(self, ini_path):
        write_config_file(ini_path, {"a": 1}, section="MAIN", encryption=None)
        write_config_file(ini_path, {"b": 2}, section="MAIN", encryption=None, mode="replace")
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out == {"b": 2}

    def test_none_section_writes_to_main(self, ini_path):
        write_config_file(ini_path, {"a": 1}, section=None, encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out == {"a": 1}

    def test_a_variable_absent_from_the_values_is_warned_and_skipped(self, ini_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert write_config_file(ini_path, {"a": 1}, section="MAIN", variables="a,nope", encryption=None) is True
        assert any("No variable nope" in rec.getMessage() for rec in caplog.records)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption=None)
        assert out == {"a": 1}


class TestEncryptionModes:
    def test_the_deprecated_xor_name_still_round_trips_and_warns(self, ini_path, caplog):
        with caplog.at_level(logging.WARNING):
            assert write_config_file(ini_path, {"k": "v"}, section="MAIN", encryption="xor") is True
        assert any("misnomer" in rec.getMessage() for rec in caplog.records)
        out: dict = {}
        assert read_config_file(ini_path, out, section="MAIN", encryption="xor") is True
        assert out == {"k": "v"}

    def test_a_value_that_is_not_base64_is_read_back_raw(self, ini_path):
        write_config_file(ini_path, {"note": "not base64 at all!"}, section="MAIN", encryption=None)
        out: dict = {}
        read_config_file(ini_path, out, section="MAIN", encryption="base64")
        assert out == {"note": "not base64 at all!"}
