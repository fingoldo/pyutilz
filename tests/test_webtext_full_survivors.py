"""Exact-output pins for text.strings.webtext (mutation survivors from the full module sweep)."""

from __future__ import annotations

import string
import sys
import types

import pytest

from pyutilz.text.strings import webtext as wt


class _FakeLogger:
    def __init__(self):
        self.errors: list = []

    def error(self, message, *args):
        self.errors.append(message)


class _FakeDoc:
    def __init__(self, text):
        self.sents = [f"S:{text}"]


class _FakeNlp:
    def __call__(self, text):
        return _FakeDoc(text)


class TestLazyGlobals:
    def test_the_lazy_cache_names_are_pinned(self):
        assert wt.__lazy_globals__ == frozenset({"inflect_engine", "nlp", "ascii_emojies", "unicode_emojies"})


class TestSpacySentences:
    def test_a_preloaded_model_is_used_without_loading_spacy(self, monkeypatch):
        monkeypatch.setattr(wt, "nlp", _FakeNlp())
        fake = types.ModuleType("spacy")
        fake.load = lambda name: pytest.fail("the cached model must not be reloaded")
        monkeypatch.setitem(sys.modules, "spacy", fake)
        assert wt.spacy_sent_tokenize("a") == ["S:a"]

    def test_the_small_english_model_is_loaded_once_when_missing(self, monkeypatch):
        calls: list = []
        fake = types.ModuleType("spacy")

        def load(name):
            calls.append(name)
            return _FakeNlp()

        fake.load = load
        monkeypatch.setitem(sys.modules, "spacy", fake)
        monkeypatch.setattr(wt, "nlp", None)
        assert wt.spacy_sent_tokenize("b") == ["S:b"]
        assert calls == ["en_core_web_sm"]


class TestRemoveVideos:
    def test_the_default_markers_remove_a_video_reference(self):
        assert wt.remove_videos("a [[VIDEOID:x]] b") == "a  b"

    def test_none_markers_fall_back_to_the_defaults(self):
        assert wt.remove_videos("a [[VIDEOID:x]] b", token=None, token2=None) == "a  b"

    def test_custom_markers_are_used(self):
        assert wt.remove_videos("a <<x>>b", token="<<", token2=">>") == "a b"

    def test_a_leading_reference_is_found(self):
        assert wt.remove_videos("[[VIDEOID:x]]tail") == "tail"

    def test_consecutive_references_are_all_removed(self):
        assert wt.remove_videos("[[VIDEOID:a]][[VIDEOID:b]]") == ""

    def test_a_reference_after_an_earlier_closing_marker_is_removed_from_its_own_span(self):
        assert wt.remove_videos("abc]] [[VIDEOID:v]]") == "abc]] "

    def test_an_empty_closing_marker_still_terminates(self):
        assert wt.remove_videos("]]x", token="") == "x"

    def test_an_unclosed_reference_is_logged_and_left_intact(self, monkeypatch):
        fake = _FakeLogger()
        monkeypatch.setattr(wt, "logger", fake)
        assert wt.remove_videos("a [[VIDEOID:x") == "a [[VIDEOID:x"
        assert fake.errors == ["Could not find video tag closing in %s" % "a [[VIDEOID:x"]


class TestFixDuplicateTokens:
    def test_repeated_whitespace_commas_and_dashes_collapse(self):
        assert wt.fix_duplicate_tokens("a,,b--c  d") == "a,b-c d"

    def test_a_double_space_collapses(self):
        assert wt.fix_duplicate_tokens("a  b") == "a b"

    def test_four_or_more_repeated_terminators_shrink_to_three(self):
        assert wt.fix_duplicate_tokens("wow!!!!") == "wow!!!"
        assert wt.fix_duplicate_tokens("wow?????") == "wow???"

    def test_three_repeated_terminators_are_kept(self):
        assert wt.fix_duplicate_tokens("wow!!!") == "wow!!!"

    def test_empty_text_is_returned_unchanged(self):
        assert wt.fix_duplicate_tokens("") == ""


class TestFixHtml:
    def test_br_variants_become_a_newline_and_the_text_is_stripped(self):
        assert wt.fix_html("  x<br>y  ") == "x\ny"

    def test_a_custom_linebreak_is_used(self):
        assert wt.fix_html("a<br>b", common_linebreak="|") == "a|b"

    def test_none_linebreak_falls_back_to_a_newline(self):
        assert wt.fix_html("a<br>b", common_linebreak=None) == "a\nb"

    def test_surrounding_whitespace_is_stripped_without_a_break(self):
        assert wt.fix_html("  x  ") == "x"

    def test_empty_and_none_input_are_returned_unchanged(self):
        assert wt.fix_html("") == ""
        assert wt.fix_html(None) is None


class TestParseHtml:
    def test_text_nodes_are_joined_with_the_default_separator(self):
        assert wt.parse_html("<p>a</p><p>b</p>") == "a. b"

    def test_a_custom_separator_is_used(self):
        assert wt.parse_html("<p>a</p><p>b</p>", sep="|") == "a|b"

    def test_plain_text_is_returned_as_is(self):
        assert wt.parse_html("<p>a</p>") == "a"


class TestFixQuotations:
    def test_curly_and_straight_quotes_become_an_apostrophe(self):
        assert wt.fix_quotations("“x”") == "'x'"

    def test_a_custom_quotation_is_used(self):
        assert wt.fix_quotations("“x”", common_quotation="|") == "|x|"

    def test_none_quotation_falls_back_to_an_apostrophe(self):
        assert wt.fix_quotations("“x”", common_quotation=None) == "'x'"

    def test_text_without_quotes_is_unchanged(self):
        assert wt.fix_quotations("plain") == "plain"
        assert wt.fix_quotations("") == ""


class TestFixSpacesAndPreviousCharacter:
    def test_a_space_before_a_comma_at_the_start_is_removed(self):
        assert wt.fix_spaces(" ,a") == ",a"

    def test_the_previous_non_whitespace_index_is_found_across_a_run(self):
        assert wt._previous_non_whitespace("xa   b", 5, [" "]) == 1

    def test_a_leading_run_of_whitespace_stops_at_index_zero(self):
        assert wt._previous_non_whitespace("  b", 2, [" "]) == 0


class TestFixBrokenSentences:
    def test_a_break_after_a_space_following_a_word_keeps_the_break(self):
        assert wt.fix_broken_sentences("x \nWord") == "x \nWord."

    def test_a_break_after_a_space_following_a_sentence_end_joins_with_a_space(self):
        assert wt.fix_broken_sentences("! \nWord") == "! Word."

    def test_a_break_at_the_very_start_before_a_capital_gets_a_period(self):
        assert wt.fix_broken_sentences("x\n Capital") == "x. Capital."

    def test_a_double_space_before_a_capital_is_not_rewritten(self):
        assert wt.fix_broken_sentences("word  Capital") == "word  Capital."

    def test_a_break_after_a_comma_before_a_letter_is_kept(self):
        assert wt.fix_broken_sentences("a,\nlower") == "a,\nlower."

    def test_a_question_mark_counts_as_a_sentence_end(self):
        assert wt.fix_broken_sentences("Really?\nWorld") == "Really? World."

    def test_a_trailing_break_is_kept(self):
        assert wt.fix_broken_sentences("hello\n") == "hello\n"

    def test_a_single_letter_gets_a_full_stop(self):
        assert wt.fix_broken_sentences("a") == "a."

    def test_several_breaks_before_capitals_are_each_repaired(self):
        assert wt.fix_broken_sentences("a\nBc\n Dd") == "a. Bc. Dd."
        assert wt.fix_broken_sentences("the cat\nBob went\n Home") == "the cat. Bob went. Home."
        assert wt.fix_broken_sentences("a\nBc\n5x") == "a. Bc. 5x."

    def test_a_space_break_after_a_sentence_end_is_kept(self):
        assert wt.fix_broken_sentences("end!\n Capital") == "end!\n Capital."

    def test_a_break_before_a_lowercase_word_is_joined_after_an_earlier_repair(self):
        assert wt.fix_broken_sentences("a\nbc\nDd") == "a bc. Dd."
        assert wt.fix_broken_sentences("hi.\nBob, a\n Dog") == "hi. Bob, a. Dog."

    def test_a_break_after_a_period_and_space_is_handled_in_order(self):
        assert wt.fix_broken_sentences("a.\n b c\n D") == "a.\n b c. D."


class TestMissedSpaceBetweenSentences:
    def test_a_terminator_at_the_start_is_split_from_a_letter(self):
        assert wt.fix_missed_space_between_sentences(".a") == ". a"

    def test_a_terminator_at_the_start_before_a_digit_is_split(self):
        assert wt.fix_missed_space_between_sentences(".5") == ". 5"

    def test_a_terminator_at_the_end_is_left_alone(self):
        assert wt.fix_missed_space_between_sentences("end.") == "end."


class TestTildaBullets:
    def test_a_tilda_without_a_following_space_is_kept(self):
        assert wt._strip_tilda_bullets("~great") == "~great"

    def test_a_tilda_without_a_preceding_space_is_kept(self):
        assert wt._strip_tilda_bullets("great~") == "great~"


class TestSentencize:
    def test_a_three_character_emoticon_ending_is_kept(self):
        assert wt.sentencize_text("Hi :-)") == "Hi :-)"

    def test_a_leading_digit_with_a_terminator_is_not_capitalised(self):
        assert wt.sentencize_text("1st.") == "1st."

    def test_a_sentence_already_ending_in_a_terminator_is_not_announced(self, capsys):
        assert wt.sentencize_text("Done!", desc="d", verbose=True) == "Done!"
        assert capsys.readouterr().out == ""

    def test_verbose_is_off_by_default(self, capsys):
        assert wt.sentencize_text("hello", desc="d") == "Hello."
        assert capsys.readouterr().out == ""

    def test_a_processed_sentence_is_announced_when_verbose(self, capsys):
        assert wt.sentencize_text("hello", desc="d", verbose=True) == "Hello."
        assert capsys.readouterr().out == "d sentencized: hello\n"

    def test_a_capitalised_terminated_sentence_is_announced_when_verbose(self, capsys):
        assert wt.sentencize_text("hello.", desc="d", verbose=True) == "Hello."
        assert capsys.readouterr().out == "d sentencized: hello.\n"

    def test_a_non_alphabetic_first_character_is_not_processed(self, capsys):
        assert wt.sentencize_text("1st.", desc="d", verbose=True) == "1st."
        assert capsys.readouterr().out == ""


class TestAsciiEmojies:
    def test_a_hyphenated_smiley_is_mapped_for_a_bracket_emoticon(self):
        table = wt.get_ascii_emojies()
        assert table[":)"] == "\U0001f60a"
        assert table[":-)"] == "\U0001f60a"

    def test_a_hyphenated_form_is_not_invented_for_a_letter_emoticon(self):
        assert ":-D" not in wt.get_ascii_emojies()


class TestSurvivorKillers:
    def test_a_video_reference_is_searched_for_its_closing_marker_after_the_opening_one(self):
        assert wt.remove_videos("a]]xxxxxxxx[[VIDEOID:v]]") == "a]]xxxxxxxx"

    def test_a_break_after_a_spaced_sentence_end_keeps_the_earlier_repair(self):
        assert wt.fix_broken_sentences("a\nBc. \nDd") == "a. Bc. Dd."

    def test_a_break_before_a_lowercase_word_keeps_the_earlier_repair(self):
        assert wt.fix_broken_sentences("a\nbc\nDd\nef") == "a bc. Dd ef."

    def test_a_break_after_a_terminator_keeps_the_earlier_repair(self):
        assert wt.fix_broken_sentences("a\nBc.\nDd") == "a. Bc. Dd."

    def test_a_trailing_emoticon_after_a_longer_sentence_is_not_terminated(self):
        assert wt.sentencize_text("Hello world :-)") == "Hello world :-)"

    def test_a_capitalised_sentence_missing_its_full_stop_is_announced_when_verbose(self, capsys):
        assert wt.sentencize_text("Hello", desc="d", verbose=True) == "Hello."
        assert capsys.readouterr().out == "d sentencized: Hello\n"

    def test_a_space_token_handled_directly_is_not_rewritten(self):
        assert wt._fix_linebreak_token("word  Capital", " ", ["\r\n"], ("!", ".", "?"), string.punctuation) == "word  Capital."
