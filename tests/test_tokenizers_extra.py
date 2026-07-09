"""Regression tests for pyutilz.text.tokenizers.AdvancedTokenizer.tokenize_db_reviews.

tokenize_db_reviews() opens a named (server-side) Postgres cursor via
db.safe_execute(..., cursor_name=..., return_cursor=True) and previously only
closed it (and the tqdm progress bar) after the while-loop finished normally.
Any exception raised inside the loop body skipped cur.close()/pbar.close(),
leaking the server-side cursor. Fixed by wrapping the loop in try/finally.
"""

from unittest.mock import Mock, patch

import pytest

from pyutilz.text.tokenizers import AdvancedTokenizer


def _make_tokenizer():
    # Bypass __init__ (which requires spacy/nltk models to be loaded) - the
    # method under test never touches spacy, only self.tokenize(), which we
    # also stub out below.
    tok = object.__new__(AdvancedTokenizer)
    return tok


class _FakeChunk:
    def __init__(self, title, body):
        self.title = title
        self.body = body


def test_tokenize_db_reviews_closes_cursor_and_pbar_on_exception():
    tok = _make_tokenizer()
    tok.tokenize = Mock(side_effect=RuntimeError("boom while tokenizing"))

    fake_cur = Mock()
    # First fetchmany returns one row (triggers self.tokenize -> raises),
    # a second call should never be needed since the exception propagates.
    fake_cur.fetchmany.return_value = [_FakeChunk("title", "some body text")]

    fake_pbar = Mock()

    with patch("pyutilz.text.tokenizers.db.safe_execute", return_value=fake_cur), patch(
        "pyutilz.text.tokenizers.tqdm", return_value=fake_pbar
    ), patch("pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text):
        with pytest.raises(RuntimeError, match="boom while tokenizing"):
            tok.tokenize_db_reviews(sql="select 1", tokens={})

    fake_cur.close.assert_called_once()
    fake_pbar.close.assert_called_once()


def test_tokenize_db_reviews_closes_cursor_and_pbar_on_fetch_exception():
    """Exception raised by fetchmany() itself (not just row processing) must
    also still trigger cursor/pbar cleanup."""
    tok = _make_tokenizer()
    tok.tokenize = Mock()

    fake_cur = Mock()
    fake_cur.fetchmany.side_effect = RuntimeError("server-side cursor died")

    fake_pbar = Mock()

    with patch("pyutilz.text.tokenizers.db.safe_execute", return_value=fake_cur), patch(
        "pyutilz.text.tokenizers.tqdm", return_value=fake_pbar
    ), patch("pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text):
        with pytest.raises(RuntimeError, match="server-side cursor died"):
            tok.tokenize_db_reviews(sql="select 1", tokens={})

    fake_cur.close.assert_called_once()
    fake_pbar.close.assert_called_once()


def test_tokenize_db_reviews_closes_cursor_and_pbar_on_normal_completion():
    """Baseline: the happy path must still close cursor/pbar exactly once."""
    tok = _make_tokenizer()
    tok.tokenize = Mock()

    fake_cur = Mock()
    fake_cur.fetchmany.side_effect = [
        [_FakeChunk("t1", "b1")],
        [],  # signals end of results
    ]

    fake_pbar = Mock()

    with patch("pyutilz.text.tokenizers.db.safe_execute", return_value=fake_cur), patch(
        "pyutilz.text.tokenizers.tqdm", return_value=fake_pbar
    ), patch("pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text):
        tok.tokenize_db_reviews(sql="select 1", tokens={})

    fake_cur.close.assert_called_once()
    fake_pbar.close.assert_called_once()
