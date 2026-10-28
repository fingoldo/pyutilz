"""Behaviour of ``pyutilz.core.pythonlib.filesystem``: the partitioned path takes the first characters of a file name as
nested folders (unsafe characters become the replacement, an empty name gives an empty path), file names are cleaned of
characters a filesystem rejects and capped at 255 characters, sizes are abbreviated to K, M or B, and a HashableDict
can be used as a set member or a dict key."""

from __future__ import annotations

from os import sep

import pytest

from pyutilz.core.pythonlib import filesystem


class TestPartitionedPath:
    def test_the_first_characters_become_nested_folders(self):
        assert filesystem.get_partitioned_filepath("abcdef.txt") == f"a{sep}b{sep}"

    def test_a_deeper_partition_takes_more_characters(self):
        assert filesystem.get_partitioned_filepath("abcdef.txt", depth=3) == f"a{sep}b{sep}c{sep}"

    def test_names_are_lowercased(self):
        assert filesystem.get_partitioned_filepath("AB.txt") == f"a{sep}b{sep}"

    def test_an_unsafe_character_becomes_the_replacement(self):
        assert filesystem.get_partitioned_filepath("a-b.txt", replace_char="_") == f"a{sep}_{sep}"

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


class TestHashableDict:
    def test_equal_contents_hash_equally(self):
        assert hash(filesystem.HashableDict({"a": 1})) == hash(filesystem.HashableDict({"a": 1}))

    def test_it_can_be_a_set_member(self):
        assert filesystem.HashableDict({"a": 1}) in {filesystem.HashableDict({"a": 1})}

    def test_it_still_compares_as_a_dict(self):
        assert filesystem.HashableDict({"a": 1}) == {"a": 1}
