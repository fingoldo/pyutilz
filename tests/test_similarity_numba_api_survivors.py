"""Default-argument, coverage-gate, dispatch and fallback pins for text.similarity._numba_api (mutation survivors).

The numba branch is live only when numba is installed. The pure-Python fallback branch is re-executed from a
fresh copy of the module with ``HAS_NUMBA`` forced off, so it is exercised in every environment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pyutilz.text.similarity import _numba_api as api
from pyutilz.text.similarity import _numba_kernels as kernels

needs_numba = pytest.mark.skipif(not kernels.HAS_NUMBA, reason="numba branch is only live when numba is installed")


def _fallback_namespace(monkeypatch) -> dict:
    """Execute the module source as a fresh namespace with the numba branch disabled (the ``else`` branch goes live)."""
    monkeypatch.setattr(kernels, "HAS_NUMBA", False)
    source = Path(api.__file__).read_text(encoding="utf-8")
    namespace: dict = {
        "__name__": "pyutilz.text.similarity._numba_api_fallback_probe",
        "__package__": "pyutilz.text.similarity",
        "__file__": api.__file__,
    }
    exec(compile(source, api.__file__, "exec"), namespace)
    return namespace


def _spy(monkeypatch, name: str) -> list:
    """Record each call to the module-level kernel ``name`` while still running the real one."""
    calls: list = []
    original = getattr(api, name)

    def wrapper(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(api, name, wrapper)
    return calls


SEQUENTIAL_BATCH = "_sentences_similarity_batch_core"
PARALLEL_BATCH = "_sentences_similarity_batch_parallel"
SEQUENTIAL_MATCHES = "_sentences_similarity_batch_with_matches_core"
PARALLEL_MATCHES = "_sentences_similarity_batch_with_matches_parallel"


@needs_numba
class TestNumbaFunctionDefaults:
    def test_default_threshold_keeps_a_single_letter_word_pair(self):
        assert api.sentences_similarity_numba(["A"], ["AB"]) == pytest.approx(0.75)

    def test_a_threshold_of_two_drops_the_single_letter_word_pair(self):
        assert api.sentences_similarity_numba(["A"], ["AB"], cMinLenTHreshold=2) == 0.0

    def test_default_coverage_side_accepts_a_passing_gate(self):
        assert api.sentences_similarity_numba(["AAA"], ["AAA"], required_coverage=0.5) == 1.0

    def test_default_min_word_similarity_is_one_half(self):
        assert api.sentences_similarity_numba(["SITTING"], ["KITTEN2"], required_coverage=1.0) == pytest.approx(0.5714, abs=1e-4)
        assert api.sentences_similarity_numba(["SITTING"], ["KITTEN2"], required_coverage=1.0, min_word_similarity=0.6) is None

    def test_an_empty_side_returns_none_on_either_side(self):
        assert api.sentences_similarity_numba([], ["X"]) is None
        assert api.sentences_similarity_numba(["X"], []) is None

    def test_an_empty_side_returns_before_the_token_warning_even_when_the_other_side_is_long(self, caplog):
        caplog.set_level(logging.WARNING)
        assert api.sentences_similarity_numba([], ["AAA"] * 51) is None
        assert api.sentences_similarity_numba(["AAA"] * 51, []) is None
        assert "safe threshold" not in caplog.text


@needs_numba
class TestNumbaTokenWarning:
    def test_exactly_fifty_tokens_do_not_warn(self, caplog):
        caplog.set_level(logging.WARNING)
        api.sentences_similarity_numba(["AAA"] * 50, ["AAA"])
        api.sentences_similarity_numba(["AAA"], ["AAA"] * 50)
        assert "safe threshold" not in caplog.text

    def test_fifty_one_tokens_on_the_first_side_warn_with_both_counts_and_the_threshold(self, caplog):
        caplog.set_level(logging.WARNING)
        api.sentences_similarity_numba(["AAA"] * 51, ["AAA"] * 3)
        assert "N_a=51, N_b=3 tokens, exceeding the safe threshold of 50" in caplog.text
        assert "greedy matching pass is O(w_min * N_a * N_b)" in caplog.text

    def test_fifty_one_tokens_on_the_second_side_warn(self, caplog):
        caplog.set_level(logging.WARNING)
        api.sentences_similarity_numba(["AAA"], ["AAA"] * 51)
        assert "N_a=1, N_b=51 tokens, exceeding the safe threshold of 50" in caplog.text


@needs_numba
class TestNumbaCoverageGate:
    def test_a_zero_similarity_is_returned_when_the_gate_passes(self):
        assert api.sentences_similarity_numba(["AAA"], ["ZZZ"], required_matched_words=0) == 0.0

    def test_the_tie_on_length_takes_side_a_for_the_max_check(self):
        assert api.sentences_similarity_numba(["the"], ["AAA"], required_coverage=0.5, stop_words=["the"], coverage_side="max") == 0.0

    def test_the_tie_on_length_takes_side_b_for_the_min_check(self):
        assert api.sentences_similarity_numba(["the"], ["AAA"], required_coverage=0.5, stop_words=["the"], coverage_side="min") is None

    def test_a_passing_max_check_does_not_run_the_min_check(self):
        assert api.sentences_similarity_numba(["the", "the"], ["AAA"], required_coverage=0.5, stop_words=["the"], coverage_side="max") == 0.0

    def test_side_both_runs_the_min_check_after_a_passing_max_check(self):
        assert api.sentences_similarity_numba(["the", "the"], ["AAA"], required_coverage=0.5, stop_words=["the"], coverage_side="both") is None

    def test_the_min_check_receives_min_word_similarity_and_coverage_in_order(self):
        assert api.sentences_similarity_numba(["AAA", "BBB"], ["AAA", "BBX"], min_word_similarity=0.8, required_coverage=0.6, coverage_side="min") is None


@needs_numba
class TestNumbaDispatch:
    def test_batch_defaults_to_the_sequential_kernel(self, monkeypatch):
        sequential = _spy(monkeypatch, SEQUENTIAL_BATCH)
        parallel = _spy(monkeypatch, PARALLEL_BATCH)
        api.sentences_similarity_numba_batch(["AB"], [["AB"]])
        assert sequential == [SEQUENTIAL_BATCH]
        assert parallel == []

    def test_batch_parallel_flag_selects_the_parallel_kernel(self, monkeypatch):
        sequential = _spy(monkeypatch, SEQUENTIAL_BATCH)
        parallel = _spy(monkeypatch, PARALLEL_BATCH)
        api.sentences_similarity_numba_batch(["AB"], [["AB"]], parallel=True)
        assert parallel == [PARALLEL_BATCH]
        assert sequential == []

    def test_batch_default_threshold_keeps_a_single_letter_word_pair(self):
        assert api.sentences_similarity_numba_batch(["A"], [["AB"]]) == [pytest.approx(0.75)]
        assert api.sentences_similarity_numba_batch(["A"], [["AB"]], cMinLenTHreshold=2) == [0.0]

    def test_index_default_threshold_keeps_a_single_letter_word_pair(self):
        assert api.SentenceSimilarityIndex([["AB"]]).query(["A"]) == [pytest.approx(0.75)]

    def test_index_default_min_word_similarity_is_one_half(self):
        index = api.SentenceSimilarityIndex([["KITTEN2"]], required_coverage=1.0)
        assert index.query(["SITTING"]) == [pytest.approx(0.5714, abs=1e-4)]

    def test_index_default_coverage_side_accepts_a_passing_gate(self):
        assert api.SentenceSimilarityIndex([["AAA"]], required_coverage=0.5).query(["AAA"]) == [1.0]

    def test_index_tie_on_length_takes_the_query_for_the_max_check(self):
        index = api.SentenceSimilarityIndex([["AAA"]], required_coverage=0.5, stop_words=["the"], coverage_side="max")
        assert index.query(["the"]) == [0.0]

    def test_index_tie_on_length_takes_the_candidate_for_the_min_check(self):
        index = api.SentenceSimilarityIndex([["the"]], required_coverage=0.5, stop_words=["the"], coverage_side="min")
        assert index.query(["AAA"]) == [0.0]

    def test_index_side_both_fails_when_the_min_check_fails(self):
        index = api.SentenceSimilarityIndex([["AAA"]], required_coverage=0.5, stop_words=["the"], coverage_side="both")
        assert index.query(["the", "the"]) == [None]

    def test_index_min_check_receives_min_word_similarity_and_coverage_in_order(self):
        index = api.SentenceSimilarityIndex([["AAA", "BBX"]], required_coverage=0.6, min_word_similarity=0.8, coverage_side="min")
        assert index.query(["AAA", "BBB"]) == [None]

    def test_index_without_coverage_uses_the_sequential_kernel(self, monkeypatch):
        sequential = _spy(monkeypatch, SEQUENTIAL_BATCH)
        matches = _spy(monkeypatch, SEQUENTIAL_MATCHES)
        api.SentenceSimilarityIndex([["AB"]]).query(["AB"])
        assert sequential == [SEQUENTIAL_BATCH]
        assert matches == []

    def test_index_with_parallel_and_no_coverage_uses_the_parallel_kernel(self, monkeypatch):
        parallel = _spy(monkeypatch, PARALLEL_BATCH)
        api.SentenceSimilarityIndex([["AB"]], parallel=True).query(["AB"])
        assert parallel == [PARALLEL_BATCH]

    def test_index_with_coverage_uses_the_sequential_matches_kernel_by_default(self, monkeypatch):
        matches = _spy(monkeypatch, SEQUENTIAL_MATCHES)
        parallel = _spy(monkeypatch, PARALLEL_MATCHES)
        api.SentenceSimilarityIndex([["AB"]], required_coverage=0.5).query(["AB"])
        assert matches == [SEQUENTIAL_MATCHES]
        assert parallel == []

    def test_index_with_coverage_and_parallel_uses_the_parallel_matches_kernel(self, monkeypatch):
        parallel = _spy(monkeypatch, PARALLEL_MATCHES)
        api.SentenceSimilarityIndex([["AB"]], parallel=True, required_coverage=0.5).query(["AB"])
        assert parallel == [PARALLEL_MATCHES]


class TestFallbackBranch:
    """The pure-Python fallback, executed from a copy of the module with the numba branch disabled."""

    def test_fallback_default_threshold_keeps_a_single_letter_word_pair(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba"](["A"], ["AB"]) == pytest.approx(0.75)
        assert fb["sentences_similarity_numba"](["A"], ["AB"], cMinLenTHreshold=2) == 0.0

    def test_fallback_default_coverage_side_accepts_a_passing_gate(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba"](["AAA"], ["AAA"], required_coverage=0.5) == 1.0

    def test_fallback_default_min_word_similarity_is_one_half(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba"](["SITTING"], ["KITTEN2"], required_coverage=1.0) == pytest.approx(0.5714, abs=1e-4)

    def test_fallback_passes_the_sentences_in_their_own_order_for_the_coverage_tie(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba"](["the"], ["AAA"], required_coverage=0.5, stop_words=["the"]) == 0.0

    def test_fallback_packed_default_threshold(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba_packed"](fb["pack_sentence"](["A"]), fb["pack_sentence"](["AB"])) == pytest.approx(0.75)

    def test_fallback_packed_returns_a_score_only_when_both_sides_are_present(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        pack = fb["pack_sentence"]
        assert fb["sentences_similarity_numba_packed"](pack(["A"]), pack(["A"])) == 1.0
        assert fb["sentences_similarity_numba_packed"](None, pack(["A"])) is None
        assert fb["sentences_similarity_numba_packed"](pack(["A"]), None) is None

    def test_fallback_pack_sentence_keeps_a_nonempty_list_and_drops_an_empty_one(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["pack_sentence"](["A"]) == ("A",)
        assert fb["pack_sentence"]([]) is None

    def test_fallback_batch_default_threshold(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["sentences_similarity_numba_batch"](["A"], [["AB"]]) == [pytest.approx(0.75)]
        assert fb["sentences_similarity_numba_batch"](["A"], [["AB"]], cMinLenTHreshold=2) == [0.0]

    def test_fallback_index_default_threshold(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["SentenceSimilarityIndex"]([["AB"]]).query(["A"]) == [pytest.approx(0.75)]

    def test_fallback_index_default_min_word_similarity_is_one_half(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        index = fb["SentenceSimilarityIndex"]([["KITTEN2"]], required_coverage=1.0)
        assert index.query(["SITTING"]) == [pytest.approx(0.5714, abs=1e-4)]

    def test_fallback_index_default_coverage_side_accepts_a_passing_gate(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        assert fb["SentenceSimilarityIndex"]([["AAA"]], required_coverage=0.5).query(["AAA"]) == [1.0]

    def test_fallback_index_query_keeps_the_query_first_for_the_coverage_tie(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        index = fb["SentenceSimilarityIndex"]([["AAA"]], required_coverage=0.5, stop_words=["the"])
        assert index.query(["the"]) == [0.0]

    def test_fallback_index_validates_the_side_only_when_coverage_is_requested(self, monkeypatch):
        fb = _fallback_namespace(monkeypatch)
        cls = fb["SentenceSimilarityIndex"]
        cls([["AAA"]], coverage_side="bogus")
        with pytest.raises(ValueError):
            cls([["AAA"]], required_coverage=0.5, coverage_side="bogus")
        with pytest.raises(ValueError):
            cls([["AAA"]], required_matched_words=1, coverage_side="bogus")
