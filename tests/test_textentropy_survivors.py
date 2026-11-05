"""Tokenizer, Markov-statistics and entropy pins for text.strings.textentropy (mutation survivors)."""

from __future__ import annotations

from collections import Counter

import pytest

from pyutilz.text.strings import textentropy


class TestTokenizeText:
    def test_strip_and_lowercase_are_applied_before_the_tokenizer(self):
        assert list(textentropy.tokenize_text("  AB  ", lambda s: list(s))) == ["a", "b"]

    def test_disabled_flags_leave_the_source_untouched(self):
        assert list(textentropy.tokenize_text("  Ab ", lambda s: list(s), lowercase=False, strip=False)) == [" ", " ", "A", "b", " "]

    def test_lowercase_alone_keeps_surrounding_spaces(self):
        assert list(textentropy.tokenize_text("  AB ", lambda s: list(s), strip=False)) == [" ", " ", "a", "b", " "]


class TestTokenizeSource:
    def test_a_file_is_read_line_by_line_with_a_trailing_space_per_line(self, tmp_path):
        path = tmp_path / "src.txt"
        path.write_text("ab\ncd\n", encoding="utf-8")
        assert list(textentropy.tokenize_to_chars(str(path), is_file=True)) == ["a", "b", " ", "c", "d", " "]

    def test_a_string_is_tokenized_without_a_trailing_space(self):
        assert list(textentropy.tokenize_to_chars("ab")) == ["a", "b"]

    def test_tokenize_source_treats_its_argument_as_text_by_default(self):
        assert list(textentropy.tokenize_source("ab", lambda s: list(s))) == ["a", "b"]


class TestTokenizeToWords:
    def test_words_are_letters_and_apostrophes_lowercased(self):
        assert list(textentropy.tokenize_to_words("Don't STOP 123 now")) == ["don't", "stop", "now"]

    def test_adjacent_non_letters_do_not_produce_empty_words(self):
        assert list(textentropy.tokenize_to_words("a,,b")) == ["a", "b"]


class TestEntropyStats:
    def test_the_markov_prefix_counts_follow_the_stream(self):
        conditional, stats = textentropy.get_entropy_stats("abab", 1)
        assert dict(stats) == {("a",): 2, ("b",): 1}
        assert conditional[("a",)]["b"] == 2

    def test_the_default_model_order_is_two_token_prefixes(self):
        _, stats = textentropy.get_entropy_stats("abcd")
        assert all(len(prefix) == 2 for prefix in stats)

    def test_a_stream_shorter_than_the_order_gives_no_stats(self):
        _, stats = textentropy.get_entropy_stats("a", 2)
        assert len(stats) == 0


class TestEntropyValues:
    def test_two_equiprobable_symbols_carry_one_bit(self):
        assert textentropy.entropy(Counter({"a": 1, "b": 1}), 2) == pytest.approx(1.0)

    def test_a_single_symbol_has_zero_entropy_and_no_negative_zero(self):
        value = textentropy.entropy(Counter({"a": 1}))
        assert value == 0.0
        assert str(value) == "0.0"

    def test_the_normalisation_factor_turns_counts_into_probabilities(self):
        assert textentropy.entropy(Counter({"a": 2, "b": 2}), 4) == pytest.approx(1.0)

    def test_zero_counts_are_skipped_not_fatal(self):
        assert textentropy.entropy(Counter({"a": 0, "b": 2}), 2) == pytest.approx(0.0)

    def test_entropy_rate_of_a_deterministic_chain_is_zero(self):
        conditional, stats = textentropy.get_entropy_stats("abab", 1)
        assert textentropy.entropy_rate(conditional, stats) == pytest.approx(0.0)


class TestComputeEntropyStats:
    def test_empty_text_gives_none_and_none(self):
        assert textentropy.compute_entropy_stats("") == (None, None)

    def test_order_zero_on_two_characters(self):
        raw, rate = textentropy.compute_entropy_stats("ab", order=0)
        assert raw == pytest.approx(0.0)
        assert rate == pytest.approx(1.0)


class TestNaiveEntropyRate:
    def test_empty_string_is_zero(self):
        assert textentropy.naive_entropy_rate("") == 0.0

    def test_two_equiprobable_characters_carry_one_bit(self):
        assert textentropy.naive_entropy_rate("ab") == pytest.approx(1.0)

    def test_a_constant_string_has_positive_zero_entropy(self):
        value = textentropy.naive_entropy_rate("aaaa")
        assert value == 0.0
        assert str(value) == "0.0"


class TestStringifyDict:
    def test_entries_are_key_equals_value_joined_by_comma(self):
        assert textentropy.stringify_dict({"a": 1, "b": 2}) == "a=1,b=2"

    def test_a_custom_separator_is_used_between_entries(self):
        assert textentropy.stringify_dict({"a": 1, "b": 2}, sep=";") == "a=1;b=2"
