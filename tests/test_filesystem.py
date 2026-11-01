"""Behaviour of ``pyutilz.core.pythonlib.filesystem``: the partitioned path takes the first characters of a file name as
nested folders (unsafe characters become the replacement, an empty name gives an empty path), file names are cleaned of
characters a filesystem rejects and capped at 255 characters, sizes are abbreviated to K, M or B, a HashableDict
can be used as a set member or a dict key, pickles load only with a matching sha256 sidecar, objects are dumped to and
loaded from one file each, the shelve helper persists values under a lock, and the stdout redirect discards output."""

from __future__ import annotations

import dbm
import errno
import logging
import os
import sys
from os import sep

import joblib
import pandas as pd
import pytest

from pyutilz.core.pythonlib import filesystem
from pyutilz.core.safe_pickle import PickleVerificationError, safe_dump, write_sidecar


class TestPartitionedPath:
    def test_the_first_characters_become_nested_folders(self):
        assert filesystem.get_partitioned_filepath("abcdef.txt") == f"a{sep}b{sep}"

    def test_a_deeper_partition_takes_more_characters(self):
        assert filesystem.get_partitioned_filepath("abcdef.txt", depth=3) == f"a{sep}b{sep}c{sep}"

    def test_names_are_lowercased(self):
        assert filesystem.get_partitioned_filepath("AB.txt") == f"a{sep}b{sep}"

    def test_an_unsafe_character_becomes_the_replacement(self):
        assert filesystem.get_partitioned_filepath("a-b.txt", replace_char="_") == f"a{sep}_{sep}"

    def test_an_unsafe_character_becomes_an_underscore_by_default(self):
        assert filesystem.get_partitioned_filepath("a/bc") == f"a{sep}_{sep}"

    def test_an_empty_name_gives_an_empty_path_not_a_root_path(self):
        assert filesystem.get_partitioned_filepath("") == ""


class TestValidFilename:
    def test_characters_a_filesystem_rejects_are_replaced(self):
        assert filesystem.ensure_valid_filename("a/b:c*d?.txt") == "a_b_c_d_.txt"

    def test_a_valid_name_is_unchanged(self):
        assert filesystem.ensure_valid_filename("report_2026.csv") == "report_2026.csv"

    def test_a_long_name_is_capped_at_the_maximum_length(self):
        assert len(filesystem.ensure_valid_filename("x" * 300)) == 255

    def test_the_cap_can_be_set_explicitly(self):
        assert filesystem.ensure_valid_filename("x" * 20, max_length=8) == "x" * 8

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("CON", "_"),
            ("con.txt", "_.txt"),
            ("CONSOLE", "CONSOLE"),
            ("xCON", "xCON"),
            ("aCOM1", "aCOM1"),
            ("COM1", "_"),
            ("LPT9", "_"),
            ("NUL.x", "_.x"),
            (" a", "_a"),
            ("  a", "_ a"),
            ("a ", "a_"),
            ("a.", "a_"),
            ("a.b", "a.b"),
            ("a b", "a b"),
        ],
    )
    def test_reserved_names_and_edge_characters(self, name, expected):
        assert filesystem.ensure_valid_filename(name) == expected

    def test_the_name_is_truncated_before_the_characters_are_replaced(self):
        assert filesystem.ensure_valid_filename("a/" * 200, max_length=3) == "a_a"


class TestHumanReadableSize:
    @pytest.mark.parametrize(
        "size,expected",
        [
            (12, "12"),
            (1500, "1.5K"),
            (2_000_000, "2.0M"),
            (3_000_000_000, "3.0B"),
        ],
    )
    def test_sizes_are_abbreviated_by_magnitude(self, size, expected):
        assert filesystem.get_human_readable_set_size(size) == expected

    def test_the_rounding_argument_sets_the_decimal_places(self):
        assert filesystem.get_human_readable_set_size(2_000_000, rounding=2) == "2.0M"
        assert filesystem.get_human_readable_set_size(1_234_567, rounding=2) == "1.23M"

    def test_a_thousand_is_the_first_kilo_value(self):
        assert filesystem.get_human_readable_set_size(999) == "999"
        assert filesystem.get_human_readable_set_size(1000) == "1.0K"


class TestHashableDict:
    def test_equal_contents_hash_equally(self):
        assert hash(filesystem.HashableDict({"a": 1})) == hash(filesystem.HashableDict({"a": 1}))

    def test_it_can_be_a_set_member(self):
        assert filesystem.HashableDict({"a": 1}) in {filesystem.HashableDict({"a": 1})}

    def test_it_still_compares_as_a_dict(self):
        assert filesystem.HashableDict({"a": 1}) == {"a": 1}

    def test_equal_dicts_with_equal_value_strings_hash_equally_in_any_order(self):
        first = filesystem.HashableDict({"a": "x", "b": "x"})
        second = filesystem.HashableDict({"b": "x", "a": "x"})
        assert first == second
        assert hash(first) == hash(second)


class TestLoadFile:
    def test_a_missing_file_raises_with_its_errno_message_and_name(self, tmp_path, caplog):
        missing = tmp_path / "nope.pckl"
        with caplog.at_level(logging.WARNING, logger=filesystem.logger.name):
            with pytest.raises(FileNotFoundError) as info:
                filesystem.load_file(str(missing))
        assert info.value.errno == errno.ENOENT
        assert info.value.strerror == os.strerror(errno.ENOENT)
        assert info.value.filename == os.path.abspath(missing)
        assert [r.getMessage() for r in caplog.records] == [f"File {os.path.abspath(missing)} not found!"]

    def test_an_unknown_extension_is_named_in_the_error(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("x")
        with pytest.raises(ValueError, match=r"^load_file: unsupported extension for "):
            filesystem.load_file(str(path))

    def test_a_joblib_file_with_a_sidecar_loads(self, tmp_path):
        path = tmp_path / "obj.joblib"
        joblib.dump({"a": 1}, path)
        write_sidecar(str(path))
        assert filesystem.load_file(str(path)) == {"a": 1}

    def test_the_extension_is_matched_regardless_of_case(self, tmp_path):
        path = tmp_path / "OBJ.JOBLIB"
        joblib.dump({"a": 2}, path)
        write_sidecar(str(path))
        assert filesystem.load_file(str(path)) == {"a": 2}

    def test_a_pickle_without_a_sidecar_is_refused_with_the_fix_named(self, tmp_path):
        path = tmp_path / "obj.joblib"
        joblib.dump([1], path)
        with pytest.raises(PickleVerificationError) as info:
            filesystem.load_file(str(path))
        message = str(info.value)
        assert "refusing to unpickle" in message
        assert "sha256 sidecar missing or mismatch" in message
        assert "write_sidecar" in message

    def test_a_pickle_changed_after_its_sidecar_was_written_is_refused(self, tmp_path):
        path = tmp_path / "obj.joblib"
        joblib.dump([1], path)
        write_sidecar(str(path))
        joblib.dump([2], path)
        with pytest.raises(PickleVerificationError):
            filesystem.load_file(str(path))

    def test_allow_unverified_loads_a_legacy_file_without_a_sidecar(self, tmp_path):
        path = tmp_path / "legacy.joblib"
        joblib.dump({"a": 3}, path)
        assert filesystem.load_file(str(path), allow_unverified=True) == {"a": 3}

    def test_a_pckl_file_is_read_as_a_pandas_object_by_default(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "f.pckl"
        frame.to_pickle(path)
        write_sidecar(str(path))
        pd.testing.assert_frame_equal(filesystem.load_file(str(path)), frame)

    def test_a_pckl_file_loads_through_the_safe_loader_when_not_read_as_pandas(self, tmp_path):
        path = tmp_path / "plain.pckl"
        safe_dump({"a": 4}, str(path))
        assert filesystem.load_file(str(path), unpickle_to_pd=False) == {"a": 4}


class TestObjectsDumper:
    def test_each_named_object_is_written_to_its_own_file(self, tmp_path):
        done = filesystem.ObjectsDumper().process_objects("a b", container={"a": [1], "b": {"x": 2}}, path=str(tmp_path))
        assert done == 2
        assert joblib.load(tmp_path / "a.pckl") == [1]
        assert joblib.load(tmp_path / "b.pckl") == {"x": 2}

    def test_a_custom_names_separator_splits_the_names(self, tmp_path):
        assert filesystem.ObjectsDumper().process_objects("a,b", container={"a": [1], "b": [2]}, path=str(tmp_path), names_sep=",") == 2

    def test_no_names_dumps_every_key_of_the_container(self, tmp_path):
        assert filesystem.ObjectsDumper().process_objects(container={"a": [1], "b": [2]}, path=str(tmp_path)) == 2
        assert (tmp_path / "b.pckl").exists()

    def test_a_namespace_prefixes_the_file_name_with_an_underscore_by_default(self, tmp_path):
        filesystem.ObjectsDumper().process_objects("a", container={"a": [1]}, path=str(tmp_path), namespace="ns")
        assert (tmp_path / "ns_a.pckl").exists()

    def test_a_custom_namespace_separator_and_extension_are_used(self, tmp_path):
        filesystem.ObjectsDumper().process_objects("a", container={"a": [1]}, path=str(tmp_path), namespace="ns", namespace_sep="-", file_extension=".joblib")
        assert (tmp_path / "ns-a.joblib").exists()

    def test_a_falsy_object_is_skipped_counted_out_and_logged(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=filesystem.logger.name):
            done = filesystem.ObjectsDumper().process_objects("a b", container={"a": [], "b": [1]}, path=str(tmp_path))
        assert done == 1
        assert not (tmp_path / "a.pckl").exists()
        assert [r.getMessage() for r in caplog.records] == ["Skipped object a."]

    def test_a_quiet_run_does_not_log_skipped_objects(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=filesystem.logger.name):
            filesystem.ObjectsDumper().process_objects("a", container={"a": []}, path=str(tmp_path), verbose=False)
        assert caplog.records == []

    def test_an_existing_file_is_rewritten_by_default(self, tmp_path):
        joblib.dump([0], tmp_path / "a.pckl")
        filesystem.ObjectsDumper().process_objects("a", container={"a": [1]}, path=str(tmp_path))
        assert joblib.load(tmp_path / "a.pckl") == [1]

    def test_an_existing_file_is_kept_when_rewriting_is_off(self, tmp_path):
        joblib.dump([0], tmp_path / "a.pckl")
        done = filesystem.ObjectsDumper(rewrite_existing=False).process_objects("a", container={"a": [1]}, path=str(tmp_path))
        assert done == 0
        assert joblib.load(tmp_path / "a.pckl") == [0]

    def test_a_missing_file_is_written_even_when_rewriting_is_off(self, tmp_path):
        done = filesystem.ObjectsDumper(rewrite_existing=False).process_objects("a", container={"a": [1]}, path=str(tmp_path))
        assert done == 1

    def test_the_default_compression_is_level_nine(self):
        assert filesystem.ObjectsDumper().process_kwargs == {"compress": 9}

    def test_the_writer_gets_the_object_then_the_file_path_and_the_kwargs(self, tmp_path):
        calls = []

        def writer(obj, fname, **kwargs):
            calls.append((obj, os.path.basename(fname), kwargs))

        filesystem.ObjectsDumper(process_fcn=writer, process_kwargs={"x": 1}).process_objects("a", container={"a": [1]}, path=str(tmp_path))
        assert calls == [([1], "a.pckl", {"x": 1})]


class TestObjectsLoader:
    def test_a_missing_or_empty_key_is_filled_from_disk(self, tmp_path):
        joblib.dump([1], tmp_path / "a.pckl")
        joblib.dump([2], tmp_path / "b.pckl")
        container = {"b": []}
        done = filesystem.ObjectsLoader().process_objects("a b", container=container, path=str(tmp_path))
        assert done == 2
        assert container == {"a": [1], "b": [2]}

    def test_a_none_value_is_filled_from_disk(self, tmp_path):
        joblib.dump([1], tmp_path / "a.pckl")
        container = {"a": None}
        filesystem.ObjectsLoader().process_objects("a", container=container, path=str(tmp_path))
        assert container == {"a": [1]}

    def test_a_non_empty_value_is_kept_by_default(self, tmp_path):
        joblib.dump([9], tmp_path / "a.pckl")
        container = {"a": [1]}
        done = filesystem.ObjectsLoader().process_objects("a", container=container, path=str(tmp_path))
        assert done == 0
        assert container == {"a": [1]}

    def test_rewriting_on_overwrites_a_non_empty_value(self, tmp_path):
        joblib.dump([9], tmp_path / "a.pckl")
        container = {"a": [1]}
        done = filesystem.ObjectsLoader(rewrite_existing=True).process_objects("a", container=container, path=str(tmp_path))
        assert done == 1
        assert container == {"a": [9]}

    def test_a_missing_file_leaves_the_container_alone(self, tmp_path):
        container = {"a": []}
        assert filesystem.ObjectsLoader().process_objects("a", container=container, path=str(tmp_path)) == 0
        assert container == {"a": []}

    def test_the_reader_gets_the_file_path_and_the_kwargs(self, tmp_path):
        joblib.dump([1], tmp_path / "a.pckl")

        def reader(fname, **kwargs):
            return (os.path.basename(fname), kwargs)

        container = {}
        filesystem.ObjectsLoader(process_fcn=reader, process_kwargs={"k": 2}).process_objects("a", container=container, path=str(tmp_path))
        assert container == {"a": ("a.pckl", {"k": 2})}

    def test_the_default_reader_takes_no_extra_arguments(self):
        assert filesystem.ObjectsLoader().process_kwargs == {}


class TestOpenSafeShelve:
    def test_a_value_written_in_the_block_is_there_after_reopening(self, tmp_path):
        db = str(tmp_path / "db")
        with filesystem.open_safe_shelve(db) as shelf:
            shelf["k"] = [1]
        with filesystem.open_safe_shelve(db, flag="r") as shelf:
            assert shelf["k"] == [1]

    def test_the_lock_file_is_created_next_to_the_database(self, tmp_path):
        with filesystem.open_safe_shelve(str(tmp_path / "db")) as shelf:
            shelf["k"] = 1
        assert (tmp_path / "db.lock").exists()

    def test_an_in_place_change_is_not_saved_without_writeback(self, tmp_path):
        db = str(tmp_path / "db")
        with filesystem.open_safe_shelve(db) as shelf:
            shelf["k"] = [1]
        with filesystem.open_safe_shelve(db) as shelf:
            shelf["k"].append(2)
        with filesystem.open_safe_shelve(db, flag="r") as shelf:
            assert shelf["k"] == [1]

    def test_an_in_place_change_is_saved_with_writeback(self, tmp_path):
        db = str(tmp_path / "db")
        with filesystem.open_safe_shelve(db) as shelf:
            shelf["k"] = [1]
        with filesystem.open_safe_shelve(db, writeback=True) as shelf:
            shelf["k"].append(2)
        with filesystem.open_safe_shelve(db, flag="r") as shelf:
            assert shelf["k"] == [1, 2]

    def test_flag_n_starts_an_empty_database(self, tmp_path):
        db = str(tmp_path / "db")
        with filesystem.open_safe_shelve(db) as shelf:
            shelf["k"] = 1
        with filesystem.open_safe_shelve(db, flag="n") as shelf:
            assert "k" not in shelf

    def test_flag_r_refuses_to_create_a_missing_database(self, tmp_path):
        with pytest.raises(dbm.error):
            with filesystem.open_safe_shelve(str(tmp_path / "missing"), flag="r"):
                pass


class TestSecondRoundBoundaries:
    def test_the_allow_unverified_argument_is_keyword_only(self, tmp_path):
        path = tmp_path / "obj.joblib"
        joblib.dump({"a": 1}, path)
        write_sidecar(str(path))
        with pytest.raises(TypeError):
            filesystem.load_file(str(path), True, True)

    def test_extra_keyword_arguments_reach_the_safe_loader(self, tmp_path):
        path = tmp_path / "plain.pckl"
        safe_dump({"a": 5}, str(path))
        assert filesystem.load_file(str(path), unpickle_to_pd=False, env_var="PYUTILZ_TEST_NOT_SET") == {"a": 5}

    def test_without_a_container_the_callers_module_globals_are_used(self, tmp_path):
        filesystem.ObjectsDumper().process_objects("FILESYSTEM_SAVED_OBJECT", path=str(tmp_path))
        assert joblib.load(tmp_path / "FILESYSTEM_SAVED_OBJECT.pckl") == [7]

    def test_the_default_rounding_keeps_one_decimal(self):
        assert filesystem.get_human_readable_set_size(1234) == "1.2K"

    def test_equal_dicts_with_int_and_str_keys_of_the_same_text_hash_equally(self):
        first = filesystem.HashableDict({1: "x", "1": "x"})
        second = filesystem.HashableDict({"1": "x", 1: "x"})
        assert first == second
        assert hash(first) == hash(second)


FILESYSTEM_SAVED_OBJECT = [7]


class TestSuppressStdoutStderr:
    def test_output_inside_the_block_is_discarded(self, capsys):
        with filesystem.suppress_stdout_stderr():
            print("hidden")
            print("oops", file=sys.stderr)
        out, err = capsys.readouterr()
        assert "hidden" not in out
        assert "oops" not in err

    def test_the_streams_are_restored_after_the_block(self):
        before_out, before_err = sys.stdout, sys.stderr
        with filesystem.suppress_stdout_stderr():
            pass
        assert sys.stdout is before_out
        assert sys.stderr is before_err

    def test_non_ascii_text_can_be_written_inside_the_block(self):
        with filesystem.suppress_stdout_stderr():
            print("Привет")

    def test_a_nested_block_restores_the_outer_block_stream(self):
        outer_before = sys.stdout
        with filesystem.suppress_stdout_stderr():
            outer_stream = sys.stdout
            with filesystem.suppress_stdout_stderr():
                pass
            assert sys.stdout is outer_stream
            print("still silent")
        assert sys.stdout is outer_before
