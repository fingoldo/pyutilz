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
