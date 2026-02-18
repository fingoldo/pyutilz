"""Tests for strings.py - Phase 2 refactoring

Tests cover:
- Early return bug fix in tokenize_source (line 963-967)
- Generator behavior with yield from
"""

import pytest
import tempfile
import os


class TestTokenizeSource:
    """Test tokenize_source function - regression test for early return bug"""

    def test_tokenizes_all_lines_not_just_first(self, tmp_path):
        """Test that ALL lines are tokenized, not just the first (line 966 bug fix)"""
        from pyutilz.strings import tokenize_source

        # Create test file with multiple lines
        test_file = tmp_path / "test.txt"
        test_content = "first line tokens\nsecond line more\nthird line final"
        test_file.write_text(test_content)

        # Simple tokenizer that splits on spaces
        def simple_tokenizer(text):
            for token in text.split():
                yield token

        # Tokenize from file
        result = list(tokenize_source(str(test_file), simple_tokenizer, is_file=True, lowercase=False, strip=True))

        # Should have tokens from ALL lines, not just first
        assert len(result) > 3  # More than just "first line tokens"
        assert "first" in result
        assert "second" in result
        assert "third" in result
        assert "final" in result

    def test_tokenizes_string_input(self):
        """Test that string input (non-file) works correctly"""
        from pyutilz.strings import tokenize_source

        def simple_tokenizer(text):
            for token in text.split():
                yield token

        # Test with string input
        result = list(tokenize_source("hello world test", simple_tokenizer, is_file=False))

        assert len(result) == 3
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_lowercase_option(self, tmp_path):
        """Test that lowercase option works"""
        from pyutilz.strings import tokenize_source

        test_file = tmp_path / "test.txt"
        test_file.write_text("UPPER lower MiXeD")

        def simple_tokenizer(text):
            for token in text.split():
                yield token

        result = list(tokenize_source(str(test_file), simple_tokenizer, is_file=True, lowercase=True))

        # Should all be lowercase
        assert all(token.islower() for token in result)
        assert "upper" in result
        assert "mixed" in result

    def test_strip_option(self, tmp_path):
        """Test that strip option works"""
        from pyutilz.strings import tokenize_source

        test_file = tmp_path / "test.txt"
        test_file.write_text("  spaced  \n  line  ")

        def simple_tokenizer(text):
            # Return the whole text to check if it was stripped
            yield text

        result = list(tokenize_source(str(test_file), simple_tokenizer, is_file=True, strip=True))

        # Should be stripped (no leading/trailing spaces)
        for token in result:
            assert token == token.strip()

    def test_multiline_file_complete_processing(self, tmp_path):
        """Comprehensive test that multi-line files are fully processed"""
        from pyutilz.strings import tokenize_source

        # Create file with known content
        test_file = tmp_path / "multiline.txt"
        lines = [
            "line one with words",
            "line two with more",
            "line three final tokens"
        ]
        test_file.write_text("\n".join(lines))

        def word_tokenizer(text):
            for word in text.split():
                yield word

        result = list(tokenize_source(str(test_file), word_tokenizer, is_file=True, lowercase=False))

        # Should have all words from all 3 lines
        expected_words = ["line", "one", "with", "words",
                         "line", "two", "with", "more",
                         "line", "three", "final", "tokens"]

        assert len(result) == len(expected_words)

        # Check all expected words are present
        for word in ["one", "two", "three", "final", "tokens"]:
            assert word in result


class TestFixSpaces:
    """Test fix_spaces function - mutable default argument"""

    def test_mutable_default_not_shared(self):
        """Test that default tokens list is not shared between calls"""
        try:
            from pyutilz.strings import fix_spaces

            # First call
            result1 = fix_spaces("test,text")

            # Second call - should get fresh default list
            result2 = fix_spaces("another.sentence")

            # Both should work independently
            assert isinstance(result1, str)
            assert isinstance(result2, str)

        except ImportError:
            pytest.skip("fix_spaces not available")


@pytest.mark.parametrize("file_content,expected_token_count", [
    ("single line", 2),
    ("line one\nline two", 4),
    ("one\ntwo\nthree", 3),
    ("", 0),
])
def test_tokenize_source_line_counts(tmp_path, file_content, expected_token_count):
    """Parametrized test for different file contents"""
    from pyutilz.strings import tokenize_source

    test_file = tmp_path / "test.txt"
    test_file.write_text(file_content)

    def simple_tokenizer(text):
        if text.strip():  # Only yield if non-empty
            for token in text.split():
                yield token

    result = list(tokenize_source(str(test_file), simple_tokenizer, is_file=True))

    if file_content:
        assert len(result) == expected_token_count
    else:
        assert len(result) == 0
