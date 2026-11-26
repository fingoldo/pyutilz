"""Behaviour of ``pyutilz.text.strings.textentropy``: tokenisers, Shannon entropy over probabilities (documented
as taking probabilities, with an optional normalisation factor), the character-frequency entropy rate, and
the dict-to-string helper. Expected values follow the functions' docstrings."""

from __future__ import annotations

from collections import Counter

import pytest

from pyutilz.text.strings.textentropy import (
    entropy,
    naive_entropy_rate,
    stringify_dict,
    tokenize_to_chars,
    tokenize_to_words,
)


class TestTokenizers:
    def test_chars_keep_every_character_including_spaces(self):
        assert list(tokenize_to_chars("Ab c")) == ["a", "b", " ", "c"]

    def test_words_are_lowercased_and_stripped_of_punctuation(self):
        assert list(tokenize_to_words("Hello, World! Hello")) == ["hello", "world", "hello"]


class TestEntropy:
    def test_two_equal_probabilities_give_one_bit(self):
        assert entropy(Counter({"a": 0.5, "b": 0.5})) == pytest.approx(1.0)

    def test_a_single_certain_symbol_gives_zero_and_not_negative_zero(self):
        result = entropy(Counter({"a": 1.0}))
        assert result == 0.0
        assert str(result) == "0.0"

    def test_four_equal_probabilities_give_two_bits(self):
        assert entropy(Counter({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})) == pytest.approx(2.0)

    def test_normalisation_factor_turns_counts_into_probabilities(self):
        assert entropy(Counter({"a": 2, "b": 2}), normalization_factor=4) == pytest.approx(1.0)

    def test_zero_counts_contribute_nothing(self):
        assert entropy(Counter({"a": 0.5, "b": 0.5, "c": 0})) == pytest.approx(1.0)


class TestNaiveEntropyRate:
    def test_two_equally_frequent_characters_give_one_bit(self):
        assert naive_entropy_rate("abab") == pytest.approx(1.0)

    def test_a_single_repeated_character_gives_zero(self):
        assert naive_entropy_rate("aaaa") == 0.0

    def test_empty_string_is_the_documented_zero_sentinel(self):
        assert naive_entropy_rate("") == 0.0


class TestStringifyDict:
    def test_default_separator_is_a_comma(self):
        assert stringify_dict({"a": 1, "b": "x"}) == "a=1,b=x"

    def test_custom_separator(self):
        assert stringify_dict({"a": 1, "b": 2}, sep=";") == "a=1;b=2"


class TestTextEntropySweepSurvivors:
    """The survivors of the textentropy sweep: the defaults of the tokenizers (lowercase, strip, not a file), the file
    branch of tokenize_source and tokenize_to_chars, the model order and prefix length of the Markov counts, the weighted
    entropy rate and its argument order, and the order-zero and empty-stream cases (textentropy.py 23-126)."""

    @staticmethod
    def _one(s):
        return [s]

    def test_the_tokenizer_strips_and_lowercases_by_default(self):
        from pyutilz.text.strings.textentropy import tokenize_text

        assert list(tokenize_text("  AbC ", self._one)) == ["abc"]
        assert list(tokenize_text("  AbC ", self._one, lowercase=False)) == ["AbC"]
        assert list(tokenize_text("  AbC ", self._one, strip=False, lowercase=False)) == ["  AbC "]

    def test_a_plain_string_is_tokenized_as_text_not_as_a_filename_by_default(self):
        from pyutilz.text.strings.textentropy import tokenize_source

        assert list(tokenize_source("  A ", self._one)) == ["a"]

    def test_a_file_is_read_line_by_line_with_its_encoding(self, tmp_path):
        from pyutilz.text.strings.textentropy import tokenize_source

        path = tmp_path / "t.txt"
        path.write_text("Été\nab\n", encoding="utf-8")
        assert list(tokenize_source(str(path), self._one, is_file=True)) == ["été", "ab"]

    def test_chars_from_a_file_get_a_trailing_space_per_line(self, tmp_path):
        from pyutilz.text.strings.textentropy import tokenize_to_chars

        path = tmp_path / "c.txt"
        path.write_text("ab\ncd\n", encoding="utf-8")
        assert list(tokenize_to_chars(str(path), is_file=True)) == ["a", "b", " ", "c", "d", " "]

    def test_chars_of_a_string_are_its_characters(self):
        from pyutilz.text.strings.textentropy import tokenize_to_chars

        assert list(tokenize_to_chars("ab")) == ["a", "b"]

    def test_the_default_model_order_is_two(self):
        from pyutilz.text.strings.textentropy import get_entropy_stats

        _cond, stats = get_entropy_stats(iter("abcd"))
        assert stats == {("a", "b"): 1, ("b", "c"): 1}

    def test_a_prefix_counts_every_time_it_recurs(self):
        from pyutilz.text.strings.textentropy import get_entropy_stats

        cond, stats = get_entropy_stats(iter("aaaa"), model_order=1)
        assert stats[("a",)] == 3
        assert cond[("a",)]["a"] == 3

    def test_the_markov_counts_pair_each_prefix_with_its_next_token(self):
        from pyutilz.text.strings.textentropy import get_entropy_stats

        cond, stats = get_entropy_stats(iter("abc"), model_order=2)
        assert dict(cond) == {("a", "b"): {"c": 1}}
        assert stats == {("a", "b"): 1}

    def test_the_weighted_entropy_rate_is_worked_out_exactly(self):
        from pyutilz.text.strings.textentropy import entropy_rate, get_entropy_stats

        cond, stats = get_entropy_stats(iter("abacxyxy"), model_order=1)
        assert entropy_rate(cond, stats) == pytest.approx(2 / 7)

    def test_an_order_zero_model_counts_every_token_under_one_empty_prefix(self):
        from pyutilz.text.strings.textentropy import compute_entropy_stats

        raw, rate = compute_entropy_stats("aab")
        assert raw == 0.0
        assert rate == pytest.approx(0.9182958340544896)

    def test_a_stream_too_short_for_its_order_has_no_stats(self):
        from pyutilz.text.strings.textentropy import compute_entropy_stats

        assert compute_entropy_stats("") == (None, None)
        assert compute_entropy_stats("a", order=1) == (None, None)

    def test_a_single_prefix_still_has_stats(self):
        from pyutilz.text.strings.textentropy import compute_entropy_stats

        raw, rate = compute_entropy_stats("aaa")
        assert raw == 0.0
        assert rate == 0.0
