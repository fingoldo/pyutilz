"""Behaviour of ``pyutilz.text.strings.webtext``: removal of embedded video markers, HTML entity unescaping,
comma spacing and the plural suffix helper. Expected values were checked against the functions' own output."""

from __future__ import annotations

import pytest

from pyutilz.text.strings.webtext import ensure_space_after_comma, remove_videos, suffixize, unescape_html


class TestRemoveVideos:
    def test_a_video_marker_is_removed_with_its_delimiters(self):
        assert remove_videos("a [[VIDEOID:x]] b") == "a  b"

    def test_text_without_markers_is_unchanged(self):
        assert remove_videos("plain text") == "plain text"


class TestUnescapeHtml:
    def test_named_entities_become_characters(self):
        assert unescape_html("a &amp; b &lt;") == "a & b <"


class TestEnsureSpaceAfterComma:
    def test_a_space_is_inserted_after_a_bare_comma(self):
        assert ensure_space_after_comma("a,b") == "a, b"


class TestSuffixize:
    @pytest.mark.parametrize("qty,expected", [(1, "apple"), (2, "apples")])
    def test_the_noun_takes_a_plural_suffix_for_more_than_one(self, qty, expected):
        assert suffixize("apple", qty) == expected
