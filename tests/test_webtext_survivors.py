"""Space and previous-character pins for text.strings.webtext (mutation survivors, first batch)."""

from __future__ import annotations

from pyutilz.text.strings import webtext as wt


class TestFixSpaces:
    def test_a_space_before_a_comma_or_full_stop_is_removed(self):
        assert wt.fix_spaces("a , b .") == "a, b."

    def test_empty_or_none_text_is_returned_unchanged(self):
        assert wt.fix_spaces("") == ""
        assert wt.fix_spaces(None) is None

    def test_custom_tokens_replace_the_default_ones(self):
        assert wt.fix_spaces("wow !", tokens=["!"]) == "wow!"
        assert wt.fix_spaces("a , b", tokens=["!"]) == "a , b"

    def test_text_without_the_space_pattern_is_left_as_is(self):
        assert wt.fix_spaces("a,b.c") == "a,b.c"


class TestPreviousNonWhitespace:
    def test_the_index_of_the_previous_character_is_returned(self):
        assert wt._previous_non_whitespace("ab c", 2, [" "]) == 1

    def test_whitespace_before_the_position_is_skipped(self):
        assert wt._previous_non_whitespace("ab  c", 4, [" "]) == 1

    def test_position_zero_has_no_previous_character(self):
        assert wt._previous_non_whitespace("abc", 0, [" "]) == -1


class TestFixBrokenSentences:
    def test_empty_and_none_are_returned_unchanged(self):
        assert wt.fix_broken_sentences("") == ""
        assert wt.fix_broken_sentences(None) is None

    def test_a_break_before_a_capital_after_a_word_becomes_a_full_stop(self):
        assert wt.fix_broken_sentences("hello\nWorld") == "hello. World."

    def test_a_break_after_a_sentence_end_becomes_a_space(self):
        assert wt.fix_broken_sentences("end.\nWorld") == "end. World."

    def test_a_break_before_a_digit_is_repaired_like_a_capital(self):
        assert wt.fix_broken_sentences("a\n5") == "a. 5"

    def test_a_break_before_a_lowercase_letter_after_a_word_becomes_a_space(self):
        assert wt.fix_broken_sentences("word\nlower") == "word lower."

    def test_a_break_before_a_space_and_capital_after_a_word_becomes_a_full_stop(self):
        assert wt.fix_broken_sentences("word\n Capital") == "word. Capital."

    def test_a_break_before_a_space_after_a_sentence_end_is_kept(self):
        assert wt.fix_broken_sentences("end!\n Capital") == "end!\n Capital."

    def test_a_trailing_letter_gets_a_full_stop(self):
        assert wt.fix_broken_sentences("finished") == "finished."

    def test_a_trailing_full_stop_is_not_doubled(self):
        assert wt.fix_broken_sentences("finished.") == "finished."
