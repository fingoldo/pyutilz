"""Comprehensive tests for pyutilz.text.similarity module.

All sentence similarity tests are parametrized across implementations:
  - sentences_similarity (pure Python)
  - sentences_similarity_numba (single-call numba)
  - sentences_similarity_numba_batch (batch numba)
  - SentenceSimilarityIndex (indexed numba)
  - parallel variants of batch and index

Includes Hypothesis property-based tests and doctest validation.
"""

import doctest
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pyutilz.text.similarity import (
    levenshtein_strings_similarity,
    contigous_strings_similarity,
    sentences_similarity,
    sentences_similarity_numba,
    sentences_similarity_numba_batch,
    sentences_similarity_numba_packed,
    SentenceSimilarityIndex,
    pack_sentence,
    normalize_sentence,
)


# ══════════════════════════════════════════════════════════════════════
# Helper: all implementations as a parametrized fixture
# ══════════════════════════════════════════════════════════════════════

def _sim_python(a, b, threshold=1):
    return sentences_similarity(a, b, cMinLenTHreshold=threshold)

def _sim_numba(a, b, threshold=1):
    return sentences_similarity_numba(a, b, cMinLenTHreshold=threshold)

def _sim_batch(a, b, threshold=1):
    results = sentences_similarity_numba_batch(a, [b], cMinLenTHreshold=threshold)
    return results[0]

def _sim_batch_parallel(a, b, threshold=1):
    results = sentences_similarity_numba_batch(a, [b], cMinLenTHreshold=threshold, parallel=True)
    return results[0]

def _sim_packed(a, b, threshold=1):
    pa = pack_sentence(a)
    pb = pack_sentence(b)
    return sentences_similarity_numba_packed(pa, pb, cMinLenTHreshold=threshold)

def _sim_index(a, b, threshold=1):
    if not b:
        idx = SentenceSimilarityIndex([[]], cMinLenTHreshold=threshold)
    else:
        idx = SentenceSimilarityIndex([b], cMinLenTHreshold=threshold)
    results = idx.query(a)
    return results[0]

def _sim_index_parallel(a, b, threshold=1):
    if not b:
        idx = SentenceSimilarityIndex([[]], cMinLenTHreshold=threshold, parallel=True)
    else:
        idx = SentenceSimilarityIndex([b], cMinLenTHreshold=threshold, parallel=True)
    results = idx.query(a)
    return results[0]


ALL_IMPLS = [
    pytest.param(_sim_python, id="python"),
    pytest.param(_sim_numba, id="numba"),
    pytest.param(_sim_batch, id="batch"),
    pytest.param(_sim_batch_parallel, id="batch_parallel"),
    pytest.param(_sim_packed, id="packed"),
    pytest.param(_sim_index, id="index"),
    pytest.param(_sim_index_parallel, id="index_parallel"),
]


# ══════════════════════════════════════════════════════════════════════
# Doctests
# ══════════════════════════════════════════════════════════════════════

def test_doctests():
    import pyutilz.text.similarity as mod
    results = doctest.testmod(mod, verbose=False)
    assert results.failed == 0, f"{results.failed} doctest(s) failed"


# ══════════════════════════════════════════════════════════════════════
# levenshtein_strings_similarity
# ══════════════════════════════════════════════════════════════════════

class TestLevenshteinStringsSimilarity:

    def test_identical(self):
        assert levenshtein_strings_similarity("hello", "hello") == 1.0

    def test_doctest_example(self):
        assert abs(levenshtein_strings_similarity("MeasureOIS21", "MeasureOIS18") - 0.8333333333333334) < 1e-10

    def test_completely_different(self):
        assert levenshtein_strings_similarity("abc", "xyz") == 0.0

    def test_one_char_diff(self):
        assert levenshtein_strings_similarity("cat", "bat") == pytest.approx(2 / 3)

    def test_different_lengths(self):
        result = levenshtein_strings_similarity("test", "testing")
        assert 0 < result < 1

    def test_empty_both(self):
        with pytest.raises(ZeroDivisionError):
            levenshtein_strings_similarity("", "")

    def test_one_empty(self):
        assert levenshtein_strings_similarity("hello", "") == 0.0

    def test_unicode(self):
        assert levenshtein_strings_similarity("привет", "привет") == 1.0
        assert levenshtein_strings_similarity("привет", "приват") == pytest.approx(5 / 6)

    def test_single_char(self):
        assert levenshtein_strings_similarity("a", "a") == 1.0
        assert levenshtein_strings_similarity("a", "b") == 0.0

    @given(s=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N"))))
    @settings(max_examples=50)
    def test_self_similarity_is_one(self, s):
        assert levenshtein_strings_similarity(s, s) == 1.0

    @given(
        a=st.text(min_size=1, max_size=15, alphabet="abcdef"),
        b=st.text(min_size=1, max_size=15, alphabet="abcdef"),
    )
    @settings(max_examples=50)
    def test_range_zero_to_one(self, a, b):
        result = levenshtein_strings_similarity(a, b)
        assert 0.0 <= result <= 1.0

    @given(
        a=st.text(min_size=1, max_size=10, alphabet="abc"),
        b=st.text(min_size=1, max_size=10, alphabet="abc"),
    )
    @settings(max_examples=50)
    def test_symmetry(self, a, b):
        assert levenshtein_strings_similarity(a, b) == levenshtein_strings_similarity(b, a)


# ══════════════════════════════════════════════════════════════════════
# contigous_strings_similarity
# ══════════════════════════════════════════════════════════════════════

class TestContigousStringsSimilarity:

    def test_identical(self):
        sim, root = contigous_strings_similarity("hello", "hello")
        assert sim == 1.0
        assert root == "hello"

    def test_doctest_1(self):
        sim, root = contigous_strings_similarity("MeosureOIS21qwe", "MeasureOIS18qwe")
        assert abs(sim - 1 / 3) < 1e-10
        assert root == "Meqwe"

    def test_doctest_2(self):
        sim, root = contigous_strings_similarity("MeosureOIS21qwe", "MeosureOIS21qwe")
        assert sim == 1.0
        assert root == "MeosureOIS21qwe"

    def test_no_common(self):
        sim, root = contigous_strings_similarity("abc", "xyz")
        assert sim == 0.0

    def test_left_only(self):
        sim, root = contigous_strings_similarity("abcXXX", "abcYYY")
        assert sim == 0.5

    def test_right_only(self):
        sim, root = contigous_strings_similarity("XXXabc", "YYYabc")
        assert sim == 0.5

    @given(s=st.text(min_size=1, max_size=20, alphabet="abcdef"))
    @settings(max_examples=30)
    def test_self_is_one(self, s):
        sim, root = contigous_strings_similarity(s, s)
        assert sim == 1.0
        assert root == s


# ══════════════════════════════════════════════════════════════════════
# normalize_sentence
# ══════════════════════════════════════════════════════════════════════

class TestNormalizeSentence:

    def test_basic(self):
        result = normalize_sentence("hello world")
        assert set(result) == {"HELLO", "WORLD"}

    def test_replaces_symbols(self):
        result = normalize_sentence("a.b,c/d-e")
        assert "A" in result and "B" in result

    def test_strips_doubled_spaces(self):
        result = normalize_sentence("a   b")
        assert "A" in result and "B" in result

    def test_abbreviations(self):
        result = normalize_sentence("FC Barcelona", abbreviations=["FC"])
        assert "FC" not in result
        assert "BARCELONA" in result

    def test_abbreviation_at_end(self):
        result = normalize_sentence("Barcelona FC", abbreviations=["FC"])
        assert "FC" not in result

    def test_empty(self):
        result = normalize_sentence("")
        assert isinstance(result, list)

    def test_returns_set_based_list(self):
        result = normalize_sentence("a a a")
        assert result == ["A"]

    def test_case_insensitive(self):
        r1 = set(normalize_sentence("Hello World"))
        r2 = set(normalize_sentence("hello world"))
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════
# sentences_similarity — parametrized across ALL implementations
# ══════════════════════════════════════════════════════════════════════

class TestSentencesSimilarityAllImpls:
    """Core tests run against every implementation."""

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_identical(self, sim_fn):
        assert sim_fn(["hello", "world"], ["hello", "world"]) == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_single_word_identical(self, sim_fn):
        assert sim_fn(["test"], ["test"]) == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_empty_a(self, sim_fn):
        assert sim_fn([], ["test"]) is None

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_empty_b(self, sim_fn):
        assert sim_fn(["test"], []) is None

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_both_empty(self, sim_fn):
        assert sim_fn([], []) is None

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_completely_different(self, sim_fn):
        result = sim_fn(["aaa", "bbb"], ["xxx", "yyy"])
        assert result is not None
        assert result < 0.3

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_one_word_match(self, sim_fn):
        result = sim_fn(["the", "cat"], ["the", "dog"])
        assert result > 0.3

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_different_lengths_penalty(self, sim_fn):
        sim_equal = sim_fn(["hello"], ["hello"])
        sim_unequal = sim_fn(["hello"], ["hello", "world"])
        assert sim_unequal < sim_equal

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_sliding_window(self, sim_fn):
        result = sim_fn(["test"], ["testing"])
        assert result is not None
        assert 0.3 < result < 1.0

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_min_length_threshold(self, sim_fn):
        result = sim_fn(["a"], ["b"], threshold=3)
        assert result is not None
        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_word_order_invariant(self, sim_fn):
        a = ["hello", "world"]
        b = ["world", "hello"]
        assert sim_fn(a, b) == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_prefix_match_bonus(self, sim_fn):
        """Word that is a prefix of another gets >0.9 similarity."""
        result = sim_fn(["test"], ["testing"])
        assert result is not None and result > 0.5

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_unicode_words(self, sim_fn):
        result = sim_fn(["ПРИВЕТ", "МИР"], ["ПРИВЕТ", "МИР"])
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_three_words_partial(self, sim_fn):
        result = sim_fn(["quick", "brown", "fox"], ["slow", "brown", "dog"])
        assert 0.2 < result < 0.8

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_many_words(self, sim_fn):
        a = ["the", "quick", "brown", "fox", "jumps"]
        b = ["the", "slow", "brown", "dog", "sits"]
        result = sim_fn(a, b)
        assert result is not None
        assert 0.2 < result < 0.8

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_single_char_words(self, sim_fn):
        result = sim_fn(["a"], ["a"])
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_real_determiner_case(self, sim_fn):
        """Real case from grammar validator fuzzy matching."""
        a = normalize_sentence("definite article specific noun")
        b = normalize_sentence("def specific")
        result = sim_fn(a, b)
        assert result is not None
        assert result > 0


# ══════════════════════════════════════════════════════════════════════
# Cross-implementation consistency (all must agree with Python)
# ══════════════════════════════════════════════════════════════════════

CONSISTENCY_CASES = [
    (["hello", "world"], ["hello", "world"]),
    (["test"], ["test"]),
    (["aaa", "bbb"], ["xxx", "yyy"]),
    (["the", "cat"], ["the", "dog"]),
    (["hello"], ["hello", "world"]),
    (["test"], ["testing"]),
    (["quick", "brown", "fox"], ["slow", "brown", "dog"]),
    (["world", "hello"], ["hello", "world"]),
    (["definite", "article"], ["def", "specific"]),
    (["ПРИВЕТ", "МИР"], ["ПРИВЕТ", "МИР"]),
    (["ABC", "DEF", "GHI"], ["ABC"]),
    (["A"], ["ABCDEF"]),
    (["LONGWORD"], ["LONG"]),
    (["X", "Y", "Z"], ["X", "Y", "Z", "W"]),
]


class TestCrossImplConsistency:
    """Every implementation must produce the exact same result as Python."""

    IMPLS_NO_PYTHON = [impl for impl in ALL_IMPLS if impl.id != "python"]

    @pytest.mark.parametrize("a,b", CONSISTENCY_CASES,
                             ids=[f"case_{i}" for i in range(len(CONSISTENCY_CASES))])
    @pytest.mark.parametrize("sim_fn", IMPLS_NO_PYTHON)
    def test_matches_python(self, sim_fn, a, b):
        py_result = _sim_python(a, b)
        other_result = sim_fn(a, b)
        if py_result is None:
            assert other_result is None
        else:
            assert other_result == pytest.approx(py_result, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Hypothesis property-based tests
# ══════════════════════════════════════════════════════════════════════

# Strategy: list of uppercase words (1-6 words, 1-10 chars each)
word_st = st.text(min_size=1, max_size=10, alphabet=st.sampled_from("ABCDEFGHIJ"))
sentence_st = st.lists(word_st, min_size=1, max_size=6)


class TestHypothesisSentenceSimilarity:

    @given(sent=sentence_st)
    @settings(max_examples=30)
    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_self_similarity_is_one(self, sim_fn, sent):
        result = sim_fn(sent, sent)
        assert result == pytest.approx(1.0)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=30)
    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_range_zero_to_one(self, sim_fn, a, b):
        result = sim_fn(a, b)
        assert result is not None
        assert 0.0 <= result <= 1.0 + 1e-10

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20)
    def test_symmetry_python(self, a, b):
        """Symmetry: sim(a,b) ≈ sim(b,a)."""
        r1 = _sim_python(a, b)
        r2 = _sim_python(b, a)
        if r1 is None:
            assert r2 is None
        else:
            assert abs(r1 - r2) < 0.01

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20)
    def test_numba_matches_python_hypothesis(self, a, b):
        """Numba must always match Python."""
        py = _sim_python(a, b)
        nb = _sim_numba(a, b)
        if py is None:
            assert nb is None
        else:
            assert nb == pytest.approx(py, abs=1e-10)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20)
    def test_batch_matches_python_hypothesis(self, a, b):
        py = _sim_python(a, b)
        ba = _sim_batch(a, b)
        if py is None:
            assert ba is None
        else:
            assert ba == pytest.approx(py, abs=1e-10)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20)
    def test_parallel_matches_python_hypothesis(self, a, b):
        py = _sim_python(a, b)
        pa = _sim_batch_parallel(a, b)
        if py is None:
            assert pa is None
        else:
            assert pa == pytest.approx(py, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Batch-specific tests
# ══════════════════════════════════════════════════════════════════════

class TestSentencesSimilarityBatch:

    def test_batch_multiple_candidates(self):
        query = ["HELLO", "WORLD"]
        candidates = [["HELLO", "WORLD"], ["FOO", "BAR"], ["HELLO", "EARTH"], ["WORLD"]]
        for parallel in (False, True):
            results = sentences_similarity_numba_batch(query, candidates, parallel=parallel)
            for i, cand in enumerate(candidates):
                expected = sentences_similarity(query, cand)
                if expected is None:
                    assert results[i] is None
                else:
                    assert results[i] == pytest.approx(expected, abs=1e-10)

    def test_empty_query(self):
        results = sentences_similarity_numba_batch([], [["a"], ["b"]])
        assert results == [None, None]

    def test_empty_candidate_in_list(self):
        results = sentences_similarity_numba_batch(["a"], [[], ["b"]])
        assert results[0] is None
        assert results[1] is not None

    def test_single_candidate(self):
        r = sentences_similarity_numba_batch(["X"], [["X"]])
        assert r[0] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# Packed-specific tests
# ══════════════════════════════════════════════════════════════════════

class TestSentencesSimilarityPacked:

    def test_none_inputs(self):
        assert sentences_similarity_numba_packed(None, pack_sentence(["x"])) is None
        assert sentences_similarity_numba_packed(pack_sentence(["x"]), None) is None

    @pytest.mark.parametrize("a,b", CONSISTENCY_CASES,
                             ids=[f"case_{i}" for i in range(len(CONSISTENCY_CASES))])
    def test_packed_matches_python(self, a, b):
        py = sentences_similarity(a, b)
        pa = pack_sentence(a)
        pb = pack_sentence(b)
        packed = sentences_similarity_numba_packed(pa, pb)
        if py is None:
            assert packed is None
        else:
            assert packed == pytest.approx(py, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Index-specific tests
# ══════════════════════════════════════════════════════════════════════

class TestSentenceSimilarityIndex:

    def test_index_matches_individual(self):
        candidates = [["HELLO", "WORLD"], ["FOO", "BAR"], ["HELLO", "EARTH"], ["WORLD"]]
        for parallel in (False, True):
            idx = SentenceSimilarityIndex(candidates, parallel=parallel)
            query = ["HELLO", "WORLD"]
            results = idx.query(query)
            for i, cand in enumerate(candidates):
                expected = sentences_similarity(query, cand)
                if expected is None:
                    assert results[i] is None
                else:
                    assert results[i] == pytest.approx(expected, abs=1e-10)

    def test_multiple_queries_same_index(self):
        candidates = [["AAA"], ["BBB"], ["CCC"]]
        idx = SentenceSimilarityIndex(candidates)
        r1 = idx.query(["AAA"])
        r2 = idx.query(["BBB"])
        assert r1[0] > r1[1]
        assert r2[1] > r2[0]

    def test_empty_query(self):
        idx = SentenceSimilarityIndex([["A"], ["B"]])
        assert idx.query([]) == [None, None]

    def test_large_index(self):
        """Index with many candidates still produces correct results."""
        candidates = [[f"WORD{i}"] for i in range(100)]
        idx = SentenceSimilarityIndex(candidates)
        results = idx.query(["WORD50"])
        assert results[50] == pytest.approx(1.0)
        assert all(r < 1.0 for i, r in enumerate(results) if i != 50 and r is not None)

    def test_parallel_matches_sequential(self):
        candidates = [["A", "B"], ["C", "D"], ["A", "C"], ["X", "Y", "Z"]]
        idx_seq = SentenceSimilarityIndex(candidates, parallel=False)
        idx_par = SentenceSimilarityIndex(candidates, parallel=True)
        query = ["A", "D"]
        r_seq = idx_seq.query(query)
        r_par = idx_par.query(query)
        for i in range(len(candidates)):
            assert r_seq[i] == pytest.approx(r_par[i], abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_one_word_vs_many(self, sim_fn):
        """Single word query against multi-word candidate."""
        result = sim_fn(["HELLO"], ["HELLO", "WORLD", "FOO", "BAR"])
        assert result is not None
        assert 0 < result < 1.0

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_very_long_word(self, sim_fn):
        """Words longer than typical."""
        long_a = "A" * 50
        long_b = "A" * 50
        result = sim_fn([long_a], [long_b])
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_mixed_lengths(self, sim_fn):
        """Mix of very short and very long words."""
        result = sim_fn(["A", "ABCDEFGHIJKLMNOP"], ["A", "ABCDEFGHIJKLMNOP"])
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_all_same_word(self, sim_fn):
        result = sim_fn(["X", "X", "X"], ["X", "X", "X"])
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_substring_words(self, sim_fn):
        """One word is substring of another — triggers sliding window."""
        result = sim_fn(["CAT"], ["CATEGORY"])
        assert result is not None
        assert result > 0.5  # prefix match bonus
