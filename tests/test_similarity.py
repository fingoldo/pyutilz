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
pytest.importorskip("jellyfish")  # transitively imported by pyutilz.text.similarity
from hypothesis import given, settings
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
        assert levenshtein_strings_similarity("", "") == 1.0

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

    def test_empty_both(self):
        sim, root = contigous_strings_similarity("", "")
        assert sim == 1.0
        assert root == ""

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

    @pytest.mark.parametrize("a,b", CONSISTENCY_CASES, ids=[f"case_{i}" for i in range(len(CONSISTENCY_CASES))])
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

    # deadline=None on the numba/joblib-backed tests below (ALL_IMPLS includes numba and
    # batch_parallel variants): Hypothesis's default 200ms deadline is incompatible with these
    # functions' one-time JIT-compilation / joblib-worker-startup cost on their FIRST call in a
    # process -- observed directly as a hypothesis.errors.FlakyFailure/DeadlineExceeded (1324ms
    # on the triggering call vs 0.19ms on Hypothesis's own reproduction re-run of the identical
    # example), not a real per-call performance regression. deadline=None is Hypothesis's own
    # documented fix for tests whose runtime is inherently variable for a reason unrelated to the
    # property being tested (see its own error message).
    @given(sent=sentence_st)
    @settings(max_examples=30, deadline=None)
    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_self_similarity_is_one(self, sim_fn, sent):
        result = sim_fn(sent, sent)
        assert result == pytest.approx(1.0)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=30, deadline=None)
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
    @settings(max_examples=20, deadline=None)  # see deadline=None note above -- _sim_numba's first-call JIT warmup
    def test_numba_matches_python_hypothesis(self, a, b):
        """Numba must always match Python."""
        py = _sim_python(a, b)
        nb = _sim_numba(a, b)
        if py is None:
            assert nb is None
        else:
            assert nb == pytest.approx(py, abs=1e-10)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20, deadline=None)  # see deadline=None note above -- _sim_batch's first-call JIT warmup
    def test_batch_matches_python_hypothesis(self, a, b):
        py = _sim_python(a, b)
        ba = _sim_batch(a, b)
        if py is None:
            assert ba is None
        else:
            assert ba == pytest.approx(py, abs=1e-10)

    @given(a=sentence_st, b=sentence_st)
    @settings(max_examples=20, deadline=None)  # see deadline=None note above -- _sim_batch_parallel's joblib worker-pool startup + JIT warmup
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

    @pytest.mark.parametrize("a,b", CONSISTENCY_CASES, ids=[f"case_{i}" for i in range(len(CONSISTENCY_CASES))])
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


class TestAsymmetricLengthNormalization:
    """Regression for a real, measured false-positive class (2026-08-12, found via autopsia's RU
    symptom-resolution fuzzy matcher): the final score used to be `res / w_min` — the greedy pass only
    ever picks `w_min` word-pairs, so a SHORT candidate needs to explain few words to average high,
    letting one lucky exact-word match dominate even when the candidate leaves most of a longer query
    unexplained. A short, wrong candidate could out-score a longer, more specific, genuinely-better
    candidate purely because it had fewer words to account for. Fixed by normalizing by `w_max`
    instead, so every unmatched word on the longer side counts as an implicit zero - a short candidate
    can no longer out-score a longer one just by leaving more of the query unaccounted for.

    Concrete real-world shape (English stand-in for the original Russian medical-phrase case): querying
    "MUSCLE WEAKNESS IN LEG" against a 2-word candidate sharing only the word "WEAKNESS"
    (["WEAKNESS", "EYELID"] — otherwise unrelated) versus a 3-word candidate matching 3 of the 4 query
    words exactly (["WEAKNESS", "LEG", "MUSCLE"]) - the longer, correct candidate must win."""

    QUERY = ["MUSCLE", "WEAKNESS", "IN", "LEG"]
    SHORT_WRONG = ["WEAKNESS", "EYELID"]
    LONG_RIGHT = ["WEAKNESS", "LEG", "MUSCLE"]

    @pytest.mark.parametrize("sim_fn", ALL_IMPLS)
    def test_longer_more_specific_candidate_beats_short_partial_match(self, sim_fn):
        short_score = sim_fn(self.QUERY, self.SHORT_WRONG)
        long_score = sim_fn(self.QUERY, self.LONG_RIGHT)
        assert long_score is not None and short_score is not None
        assert long_score > short_score, (
            f"the 3-word candidate matching 3/4 query words ({long_score:.3f}) must beat the 2-word "
            f"candidate matching only 1/4 ({short_score:.3f}) - a short partial match must never "
            "out-score a longer, more complete one"
        )

    def test_index_reproduces_the_ranking(self):
        """Same fixture through SentenceSimilarityIndex (the batched/cached path every real caller in
        this codebase actually uses, e.g. autopsia's SNOMED finding bridge / lay-synonym matcher)."""
        index = SentenceSimilarityIndex([self.SHORT_WRONG, self.LONG_RIGHT], parallel=False)
        short_score, long_score = index.query(self.QUERY)
        assert long_score > short_score

    def test_equal_length_case_matches_hand_computed_value(self):
        """Pins the exact score for an equal-length pair (w_min == w_max, this module's own documented
        primary use case - team names, addresses) so a future change to this formula cannot silently
        regress the equal-length case while only being caught by the asymmetric-length tests above.
        "QUICK"/"SLOW" share no letters worth a partial-match bonus (Levenshtein sim ~0), "BROWN" is an
        exact match (1.0), "FOX"/"DOG" share no letters either (~0) - greedy picks BROWN=BROWN first,
        then the best of the two remaining weak pairs; either way the average is dominated by the one
        real match, divided by 3 (w_min == w_max == 3 here, so the normalization change is a no-op)."""
        a, b = ["QUICK", "BROWN", "FOX"], ["SLOW", "BROWN", "DOG"]
        result = sentences_similarity(a, b)
        assert result is not None
        assert 0.2 < result < 0.5


# ══════════════════════════════════════════════════════════════════════
# Long-input safety warning (documented O(w*N^2) complexity guard)
# ══════════════════════════════════════════════════════════════════════

class TestLongInputWarning:
    """sentences_similarity/_numba log a warning above SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD
    tokens, since the greedy matching pass is O(w_min * N_a * N_b) and degrades quadratically-ish
    on long inputs (benchmarked: N=10 -> ~0.3ms, N=80 -> ~25-30ms per call, pure-Python path)."""

    @pytest.mark.parametrize("sim_fn", [sentences_similarity, sentences_similarity_numba])
    def test_warns_above_threshold(self, sim_fn, caplog):
        from pyutilz.text.similarity import SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD

        n = SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD + 1
        a = [f"WORD{i}" for i in range(n)]
        b = [f"WORD{i}" for i in range(n)]
        with caplog.at_level("WARNING"):
            sim_fn(a, b)
        assert any("exceeding the safe threshold" in r.message for r in caplog.records)

    @pytest.mark.parametrize("sim_fn", [sentences_similarity, sentences_similarity_numba])
    def test_no_warning_below_threshold(self, sim_fn, caplog):
        from pyutilz.text.similarity import SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD

        n = SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD - 1
        a = [f"WORD{i}" for i in range(n)]
        b = [f"WORD{i}" for i in range(n)]
        with caplog.at_level("WARNING"):
            sim_fn(a, b)
        assert not any("exceeding the safe threshold" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════
# Sort-based njit greedy matcher (_greedy_match_sorted) must exactly match the
# plain O(w_min*N_a*N_b) rescan in _sentences_similarity_core, including tie-breaks.
# ══════════════════════════════════════════════════════════════════════

class TestSortedGreedyMatchDifferential:
    """_greedy_match_sorted (sort-once-then-scan) must pick EXACTLY the same pairs, in the
    same order, as the plain rescan loop -- including on ties, where the rescan's
    `row[j] >= best_perf` breaks ties toward the LAST-visited (largest i, then largest j) cell.
    """

    @staticmethod
    def _random_sim_matrix(rng, n_a, n_b, decimals=None):
        import numpy as np

        mat = rng.random((n_a, n_b))
        if decimals is not None:
            mat = np.round(mat, decimals)
        return mat

    @staticmethod
    def _oldscan(sim_res, n_a, n_b, w_min):
        excluded_a = [False] * n_a
        excluded_b = [False] * n_b
        res = 0.0
        for _ in range(w_min):
            best_perf = 0.0
            best_i = 0
            best_j = 0
            for i in range(n_a):
                if excluded_a[i]:
                    continue
                for j in range(n_b):
                    if not excluded_b[j] and sim_res[i, j] >= best_perf:
                        best_perf = sim_res[i, j]
                        best_i = i
                        best_j = j
            res += sim_res[best_i, best_j]
            excluded_a[best_i] = True
            excluded_b[best_j] = True
        return res

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 20, 50])
    @pytest.mark.parametrize("decimals", [None, 2, 1], ids=["no_ties", "light_ties", "heavy_ties"])
    def test_matches_oldscan_exactly(self, n, decimals):
        from pyutilz.text.similarity import _greedy_match_sorted
        import random as _random

        rng = _random.Random(f"seed-{n}-{decimals}")

        import numpy as np

        npy_rng = np.random.default_rng(abs(hash((n, decimals))) % (2**32))
        for trial in range(5):
            n_a = n
            n_b = n if trial % 2 == 0 else max(1, n - 1)
            sim_res = self._random_sim_matrix(npy_rng, n_a, n_b, decimals)
            w_min = min(n_a, n_b)

            expected = self._oldscan(sim_res, n_a, n_b, w_min)
            actual = _greedy_match_sorted(sim_res, n_a, n_b, w_min)
            assert actual == pytest.approx(expected, abs=1e-12), f"mismatch at n_a={n_a} n_b={n_b} decimals={decimals} trial={trial}"

    def test_matches_via_full_core_dispatch(self):
        """End-to-end: sentences_similarity_numba must match sentences_similarity for inputs
        large enough to route through the sorted-dispatch path, and small ones through the
        plain rescan path (dispatch threshold is on N_a*N_b, see _SORTED_MATCH_THRESHOLD)."""
        for n in (2, 5, 20):
            a = [f"WORD{i}" for i in range(n)]
            b = [f"WORD{i}" for i in range(n)]
            expected = sentences_similarity(a, b)
            actual = sentences_similarity_numba(a, b)
            assert actual == pytest.approx(expected, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Coverage gate: min_word_similarity / required_coverage / required_matched_words / coverage_side /
# stop_words - opt-in (default off, zero overhead), covered across every implementation that
# supports it: sentences_similarity (pure Python), sentences_similarity_numba, SentenceSimilarityIndex.
# (sentences_similarity_numba_batch/_packed do not support these params - out of scope, see the
# similarity.py docstrings for the with-matches numba cores this gate is built on.)
# ══════════════════════════════════════════════════════════════════════


def _cov_python(a, b, **kw):
    return sentences_similarity(a, b, **kw)


def _cov_numba(a, b, **kw):
    return sentences_similarity_numba(a, b, **kw)


def _cov_index(a, b, **kw):
    idx = SentenceSimilarityIndex([b], parallel=False, **kw)
    return idx.query(a)[0]


COVERAGE_IMPLS = [
    pytest.param(_cov_python, id="python"),
    pytest.param(_cov_numba, id="numba"),
    pytest.param(_cov_index, id="index"),
]


class TestCoverageGate:
    """Coverage gate found and built after `TestAsymmetricLengthNormalization` above: fixing the
    w_max normalization was not sufficient on its own for a QUERY-contains-CANDIDATE shape (rather
    than two independent sentences) - a 1-word candidate fully contained in a longer query still
    "wins" by construction (100% of the CANDIDATE's own words matched), even though a real, distinct
    word in the QUERY was left completely unaccounted for. Measured live in autopsia's RU symptom
    resolver, 2026-08-12: "опущение века" (eyelid ptosis) matched bare "опущение" (a real entry for
    an unrelated concept, Pelvic organ prolapse) via exactly this shape - the dropped word "века"
    (eyelid) is what actually determined the correct answer. English stand-in used here to keep the
    fixture readable without requiring Cyrillic stemming context."""

    QUERY = ["EYELID", "DROOPING"]
    CANDIDATE_PARTIAL = ["DROOPING"]  # only 1 of 2 query words covered - the bug shape
    CANDIDATE_FULL = ["EYELID", "DROOPING"]  # both query words covered

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_default_is_off_identical_score_to_no_coverage_args(self, sim_fn):
        """Not passing required_coverage/required_matched_words must reproduce the exact pre-existing
        score - the gate is additive, never a behavior change for existing callers."""
        with_defaults = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL)
        plain = sentences_similarity(self.QUERY, self.CANDIDATE_PARTIAL)
        assert with_defaults == pytest.approx(plain, abs=1e-10)

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_full_coverage_required_refuses_a_partially_covered_query(self, sim_fn):
        """The exact ptosis-shaped bug: a candidate contained in a longer query, leaving a real query
        word unaccounted for, must be REFUSED (None) under required_coverage=1.0 against the query."""
        result = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_coverage=1.0, coverage_side="max")
        assert result is None

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_full_coverage_required_accepts_a_fully_covered_query(self, sim_fn):
        result = sim_fn(self.QUERY, self.CANDIDATE_FULL, required_coverage=1.0, coverage_side="max")
        assert result is not None
        assert result == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_coverage_side_min_checks_the_shorter_side_not_the_longer(self, sim_fn):
        """coverage_side="min" asks "is most of the CANDIDATE covered", not "is most of the QUERY
        covered" - the 1-word candidate is trivially 100% covered by itself, so this must ACCEPT the
        exact case that coverage_side="max" refuses above. Demonstrates the two are genuinely
        different guarantees, not interchangeable synonyms of "some coverage requirement"."""
        result = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_coverage=1.0, coverage_side="min")
        assert result is not None

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_coverage_side_both_refuses_if_either_side_fails(self, sim_fn):
        result = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_coverage=1.0, coverage_side="both")
        assert result is None  # max-side check still fails

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_required_matched_words_absolute_count(self, sim_fn):
        """2 non-stopword query words, only 1 achieves an above-floor match - requiring 2 must refuse,
        requiring 1 must accept."""
        assert sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_matched_words=2, coverage_side="max") is None
        assert sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_matched_words=1, coverage_side="max") is not None

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_stop_words_exempt_a_word_from_the_coverage_requirement(self, sim_fn):
        """"EYELID" is the query word that fails to match CANDIDATE_PARTIAL. Declaring it a stop word
        removes it from the coverage denominator entirely, so full coverage of the one REMAINING
        content word ("DROOPING", which does match) must now pass."""
        refused = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_coverage=1.0, coverage_side="max")
        assert refused is None
        accepted = sim_fn(self.QUERY, self.CANDIDATE_PARTIAL, required_coverage=1.0, coverage_side="max", stop_words=["EYELID"])
        assert accepted is not None

    @pytest.mark.parametrize("sim_fn", COVERAGE_IMPLS)
    def test_min_word_similarity_floor_rejects_a_weak_greedy_pairing(self, sim_fn):
        """A greedy pick is still made even between two unrelated words (it is the "least bad"
        available pair) - min_word_similarity must refuse to count that as covering the word."""
        query = ["ZEBRA", "DROOPING"]
        candidate = ["QUXQUX", "DROOPING"]  # "ZEBRA" vs "QUXQUX" greedily pair with near-zero similarity
        # A very high floor: even a weak pairing scoring above 0 but below the floor doesn't count.
        result = sim_fn(query, candidate, required_coverage=1.0, coverage_side="max", min_word_similarity=0.99)
        assert result is None

    def test_all_stopwords_query_trivially_passes(self):
        """A sentence made entirely of stopwords has nothing to require coverage of - must not refuse
        (there is no content word left unaccounted for, by definition)."""
        result = sentences_similarity(["THE", "A"], ["THE"], required_coverage=1.0, coverage_side="max", stop_words=["THE", "A"])
        assert result is not None

    def test_invalid_coverage_side_raises(self):
        with pytest.raises(ValueError):
            sentences_similarity(["A"], ["A"], required_coverage=1.0, coverage_side="bogus")
        with pytest.raises(ValueError):
            sentences_similarity_numba(["A"], ["A"], required_coverage=1.0, coverage_side="bogus")
        with pytest.raises(ValueError):
            SentenceSimilarityIndex([["A"]], required_coverage=1.0, coverage_side="bogus")

    def test_numba_and_python_paths_agree_on_a_larger_random_case(self):
        """Differential check: the numba with-matches core (_sentences_similarity_core_with_matches)
        must reach the identical accept/refuse verdict and score as the pure-Python path."""
        import random

        rng = random.Random("coverage-gate-differential")
        words = [f"W{i}" for i in range(12)]
        for _ in range(20):
            n_a = rng.randint(2, 8)
            n_b = rng.randint(2, 8)
            a = rng.choices(words, k=n_a)
            b = rng.choices(words, k=n_b)
            for req_cov in (0.5, 1.0):
                py = sentences_similarity(a, b, required_coverage=req_cov, coverage_side="max")
                nb = sentences_similarity_numba(a, b, required_coverage=req_cov, coverage_side="max")
                if py is None:
                    assert nb is None
                else:
                    assert nb is not None
                    assert nb == pytest.approx(py, abs=1e-10)

    def test_index_agrees_with_pairwise_calls_across_multiple_candidates(self):
        """SentenceSimilarityIndex's coverage-active loop must reproduce per-candidate
        sentences_similarity_numba results exactly, not just for a single candidate."""
        query = ["EYELID", "DROOPING", "SEVERE"]
        candidates = [
            ["DROOPING"],
            ["EYELID", "DROOPING"],
            ["EYELID", "DROOPING", "SEVERE"],
            ["UNRELATED", "WORDS", "ENTIRELY"],
        ]
        idx = SentenceSimilarityIndex(candidates, required_coverage=0.66, coverage_side="max")
        indexed = idx.query(query)
        direct = [sentences_similarity_numba(query, c, required_coverage=0.66, coverage_side="max") for c in candidates]
        for i, (got, expected) in enumerate(zip(indexed, direct)):
            if expected is None:
                assert got is None, f"candidate {i}"
            else:
                assert got == pytest.approx(expected, abs=1e-10), f"candidate {i}"


# ══════════════════════════════════════════════════════════════════════
# `stop_words` also strips the BASE (non-coverage) w_max-normalized score, 2026-08-16
# ══════════════════════════════════════════════════════════════════════
# The w_max length-normalization fix (2026-08-12) correctly stopped a short candidate label from
# out-scoring a longer one on one lucky word - but it also means a connective word ordinary phrasing
# carries ("some", "I", "have") counts as query length the candidate can never explain, dragging a
# genuinely correct single-word match below any reasonable threshold. Found live in autopsia's complaint
# parser: "some nausea" against "Nausea" scored 0.0 (used to score 1.0 pre-w_max-fix). `stop_words` was
# already threaded through every function below for the opt-in coverage gate - these tests pin that it
# now ALSO strips before the base score is computed, not only inside the gate.


class TestStopWordsAffectBaseScore:
    _STOP = {"I", "HAVE", "A", "MY", "SOME"}

    def test_pure_python_recovers_the_exact_match_once_the_connective_word_is_stripped(self):
        assert sentences_similarity(["SOME", "NAUSEA"], ["NAUSEA"], stop_words=self._STOP) == 1.0
        # Without stop_words, the connective word still drags the score down (regression guard for the
        # guard itself - proves the fixture actually exercises the defect, not a no-op).
        assert sentences_similarity(["SOME", "NAUSEA"], ["NAUSEA"]) < 1.0

    def test_numba_agrees_with_pure_python(self):
        py = sentences_similarity(["I", "HAVE", "NAUSEA"], ["NAUSEA"], stop_words=self._STOP)
        nb = sentences_similarity_numba(["I", "HAVE", "NAUSEA"], ["NAUSEA"], stop_words=self._STOP)
        assert py == 1.0
        assert nb == pytest.approx(py, abs=1e-10)

    def test_index_recovers_the_exact_match_and_stays_index_pairwise_consistent(self):
        candidates = [["NAUSEA"], ["ABDOMINAL", "PAIN"]]
        idx = SentenceSimilarityIndex(candidates, stop_words=self._STOP)
        indexed = idx.query(["SOME", "NAUSEA"])
        assert indexed[0] == 1.0
        direct = [sentences_similarity_numba(["SOME", "NAUSEA"], c, stop_words=self._STOP) for c in candidates]
        for got, expected in zip(indexed, direct):
            assert got == pytest.approx(expected, abs=1e-10)

    def test_a_query_of_only_stop_words_falls_back_to_the_unfiltered_query_rather_than_scoring_nothing(self):
        """`_strip_stop_words` refuses to empty a sentence entirely - a query that is ALL stop words keeps
        its original words so the caller still gets a real (if low) score, never a crash on an empty list."""
        assert sentences_similarity(["I", "HAVE"], ["NAUSEA"], stop_words=self._STOP) is not None

    def test_no_stop_words_argument_is_a_true_no_op(self):
        """Zero behavior change on the default (stop_words=None) path - the same invariant this file's
        other coverage-gate tests already pin for required_coverage/min_word_similarity."""
        a, b = ["SOME", "NAUSEA"], ["NAUSEA"]
        assert sentences_similarity(a, b) == sentences_similarity(a, b, stop_words=None)
        assert sentences_similarity_numba(a, b) == sentences_similarity_numba(a, b, stop_words=None)
