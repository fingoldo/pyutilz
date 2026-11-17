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


class TestConfigFilesSweepSurvivors:
    """The survivors of the configfiles sweep: the deprecated "xor" mode name, the default MAIN section, the messages logged
    for a bad section, a failed write and the undecodable or non-literal values, and the empty-list default of the variable
    names (configfiles.py 14-181)."""

    def test_the_deprecated_xor_mode_still_decodes_base64(self, tmp_path):
        import base64

        from pyutilz.text.strings.configfiles import read_config_file

        path = tmp_path / "old.ini"
        path.write_text("[S]\nk = " + base64.b64encode(b"hello").decode() + "\n", encoding="utf-8")
        out: dict = {}
        assert read_config_file(str(path), out, section="S", encryption="xor") is True
        assert out == {"k": "hello"}

    def test_the_default_section_of_a_write_is_main(self, tmp_path):
        from pyutilz.text.strings.configfiles import write_config_file

        path = tmp_path / "default.ini"
        assert write_config_file(str(path), {"a": 1}, encryption=None) is True
        assert "[MAIN]" in path.read_text(encoding="utf-8")

    def test_a_non_string_section_is_refused_and_logged_with_its_type(self, tmp_path, caplog):
        import logging

        from pyutilz.text.strings.configfiles import read_config_file

        path = tmp_path / "x.ini"
        path.write_text("[S]\nk = 1\n", encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            assert read_config_file(str(path), {}, section=5) is None
        assert "section must be str or None, got int" in caplog.text

    def test_a_failed_write_is_logged_with_the_error_text(self, tmp_path, caplog):
        import logging

        from pyutilz.text.strings.configfiles import write_config_file

        target = tmp_path / "missing_dir" / "x.ini"
        with caplog.at_level(logging.ERROR):
            assert write_config_file(str(target), {"a": 1}) is None
        assert "No such file or directory" in caplog.text

    def test_a_value_that_is_not_base64_is_kept_with_a_debug_note(self, caplog):
        import logging

        from pyutilz.text.strings.configfiles import _maybe_b64_decode

        with caplog.at_level(logging.DEBUG):
            assert _maybe_b64_decode("abc", "key", "base64") == "abc"
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(m.startswith("Value for key is not base64-encoded, using raw string: ") for m in messages)

    def test_a_value_that_is_not_a_literal_is_kept_with_a_debug_note(self, caplog):
        import logging

        from pyutilz.text.strings.configfiles import _literal_or_raw

        with caplog.at_level(logging.DEBUG):
            assert _literal_or_raw("not a literal", "key") == "not a literal"
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(m.startswith("Value for key is not a Python literal, keeping as string: ") for m in messages)

    def test_no_variable_list_means_every_key_of_the_object(self):
        from pyutilz.text.strings.configfiles import _normalize_variables

        assert _normalize_variables([], {"a": 1, "b": 2}) == ["a", "b"]
        assert _normalize_variables(None, {"a": 1}) == ["a"]

    def test_a_non_list_of_variables_is_refused_with_its_type(self):
        from pyutilz.text.strings.configfiles import _normalize_variables

        with pytest.raises(AssertionError, match="expected list of variables, got int"):
            _normalize_variables(5, {})
