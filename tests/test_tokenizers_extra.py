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

    with patch("pyutilz.database.db.safe_execute", return_value=fake_cur), patch("tqdm.notebook.tqdm", return_value=fake_pbar), patch(
        "pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text
    ):
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

    with patch("pyutilz.database.db.safe_execute", return_value=fake_cur), patch("tqdm.notebook.tqdm", return_value=fake_pbar), patch(
        "pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text
    ):
        with pytest.raises(RuntimeError, match="server-side cursor died"):
            tok.tokenize_db_reviews(sql="select 1", tokens={})

    fake_cur.close.assert_called_once()
    fake_pbar.close.assert_called_once()


# ---------------------------------------------------------------------------
# AdvancedTokenizer.tokenize() -- morpheme-length cap (2026-07-21 audit round 2, HIGH)
# ---------------------------------------------------------------------------
# tokenize() used to enumerate EVERY substring of every word with no length cap, an
# O(word_len**3) blowup reachable on ordinary scraped-web text (a long URL, a base64/JS blob,
# any unbroken run with no whitespace) -- not just adversarial input. These tests bypass
# __init__ (which requires the spacy en_core_web_sm model, not installed here) since tokenize()
# itself never touches the spacy `nlp` global -- only nltk's sent/word tokenizers.

pytest.importorskip("nltk")

import time
from collections import defaultdict


def _make_real_tokenizer():
    """AdvancedTokenizer with its stat dicts initialized exactly as __init__ would, minus the
    spacy model load (tokenize() doesn't touch spacy)."""
    tok = object.__new__(AdvancedTokenizer)
    tok.NUM_AS_SEPARATE_WORD = defaultdict(int)
    tok.NUM_OCCS = defaultdict(int)
    tok.NUM_FIRSTLETTER_CAPITAL = defaultdict(int)
    tok.NUM_ALLLETTERS_CAPITAL = defaultdict(int)
    tok.INWORD_ABSOLUTE_POSITION = defaultdict(int)
    tok.INWORD_RELATIVE_POSITION = defaultdict(int)
    tok.NUM_FIRSTWORD_INSENTENCE = defaultdict(int)
    tok.NUM_LASTWORD_INSENTENCE = defaultdict(int)
    tok.INSENTENCE_ABSOLUTE_POSITION = defaultdict(int)
    tok.INSENTENCE_RELATIVE_POSITION = defaultdict(int)
    tok.NUM_PREV_WORDS = dict()
    tok.NUM_PREV_SENTENCE_WORDS = dict()
    return tok


class TestMorphemeLengthCap:
    def test_pathological_unbroken_word_completes_quickly(self):
        """The audit's own repro measured n=800 chars taking ~1.4s (cubic blowup). A linear-cost
        implementation should handle n=2000 in well under a second."""
        tok = _make_real_tokenizer()
        pathological = "a" * 2000

        start = time.perf_counter()
        tok.tokenize(pathological + ".")
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"tokenize() took {elapsed:.3f}s for a 2000-char word -- morpheme-length cap not applied"

    def test_ordinary_word_longer_than_morpheme_cap_still_gets_whole_word_stats(self):
        """A normal word longer than MAX_MORPHEME_LENGTH (8) but shorter than MAX_WORD_LENGTH
        (40) must still be recorded as a whole word -- the morpheme-length cap must not silently
        drop NUM_AS_SEPARATE_WORD/NUM_OCCS tracking for ordinary long words, only bound the
        substring-enumeration cost."""
        tok = _make_real_tokenizer()
        word = "understanding"  # 13 chars > MAX_MORPHEME_LENGTH=8, < MAX_WORD_LENGTH=40
        assert len(word) > AdvancedTokenizer.MAX_MORPHEME_LENGTH
        assert len(word) <= AdvancedTokenizer.MAX_WORD_LENGTH

        tok.tokenize(f"I like {word}.")

        assert tok.NUM_AS_SEPARATE_WORD[word] == 1
        assert tok.NUM_OCCS[word] >= 1

    def test_word_beyond_max_word_length_is_skipped_not_crashed(self):
        """A pathologically long word (beyond MAX_WORD_LENGTH) must be skipped entirely --
        no whole-word entry, no crash, no timeout."""
        tok = _make_real_tokenizer()
        long_word = "x" * (AdvancedTokenizer.MAX_WORD_LENGTH + 20)

        tok.tokenize(f"See {long_word} here.")

        assert long_word not in tok.NUM_AS_SEPARATE_WORD
        assert long_word not in tok.NUM_OCCS

    def test_short_words_still_chain_via_prev_words(self):
        """Ordinary short words (within the morpheme cap) must keep working exactly as before --
        the NUM_PREV_WORDS chain between consecutive short words is unaffected by the cap."""
        tok = _make_real_tokenizer()
        tok.tokenize("the cat sat.")

        assert tok.NUM_AS_SEPARATE_WORD["cat"] == 1
        assert "the" in tok.NUM_PREV_WORDS.get("cat", {})


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

    with patch("pyutilz.database.db.safe_execute", return_value=fake_cur), patch("tqdm.notebook.tqdm", return_value=fake_pbar), patch(
        "pyutilz.text.tokenizers.sentencize_text", side_effect=lambda text: text
    ):
        tok.tokenize_db_reviews(sql="select 1", tokens={})

    fake_cur.close.assert_called_once()
    fake_pbar.close.assert_called_once()
