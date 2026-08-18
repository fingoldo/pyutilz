"""Tests for the batched "with-matches" numba kernel
(`_sentences_similarity_batch_with_matches_core`/`_sentences_similarity_batch_with_matches_parallel` in
`pyutilz.text.similarity`) that `SentenceSimilarityIndex.query`'s coverage-active path now uses instead of
its former pure-Python per-candidate loop.

Reference/ground truth for correctness: `sentences_similarity_numba` (the existing single-candidate
"with-matches" path, `_compare_one_candidate_with_matches`, unchanged by this work) applied one candidate
at a time - exactly what the survey named as the correctness-first baseline the batch kernel must match.
"""

import time

import pytest

pytest.importorskip("numba")
pytest.importorskip("jellyfish")

from pyutilz.text.similarity import (
    sentences_similarity_numba,
    SentenceSimilarityIndex,
)

QUERY = ["PATIENT", "HAS", "SEVERE", "HEADACHE"]

CANDIDATES = [
    ["SEVERE", "HEADACHE"],
    ["MILD", "HEADACHE"],
    ["PATIENT", "HAS", "SEVERE", "HEADACHE"],
    ["NAUSEA"],
    ["SEVERE", "MIGRAINE"],
    [],
    ["HEADACHE"],
    ["PATIENT", "FEELS", "SEVERE", "PAIN"],
    ["THROBBING", "HEADACHE", "WITH", "AURA"],
    ["SEVERE", "HEADACHE", "AND", "NAUSEA"],
]


def _reference(query, candidates, **kw):
    """One-candidate-at-a-time via the existing, unchanged single-query with-matches path."""
    return [sentences_similarity_numba(query, c, **kw) if c else None for c in candidates]


class TestBatchWithMatchesKernelCorrectness:
    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("coverage_side", ["max", "min", "both"])
    def test_matches_single_candidate_reference(self, parallel, coverage_side):
        kw = dict(required_coverage=0.5, coverage_side=coverage_side)
        idx = SentenceSimilarityIndex(CANDIDATES, parallel=parallel, **kw)
        got = idx.query(QUERY)
        expected = _reference(QUERY, CANDIDATES, **kw)
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            if e is None:
                assert g is None
            else:
                assert g == pytest.approx(e, abs=1e-9)

    def test_required_matched_words_matches_reference(self):
        kw = dict(required_matched_words=2, coverage_side="max")
        idx = SentenceSimilarityIndex(CANDIDATES, parallel=True, **kw)
        got = idx.query(QUERY)
        expected = _reference(QUERY, CANDIDATES, **kw)
        for g, e in zip(got, expected):
            assert (g is None) == (e is None)
            if e is not None:
                assert g == pytest.approx(e, abs=1e-9)

    def test_stop_words_matches_reference(self):
        kw = dict(required_coverage=1.0, coverage_side="max", stop_words=["HAS", "AND", "WITH"])
        idx = SentenceSimilarityIndex(CANDIDATES, parallel=True, **kw)
        got = idx.query(QUERY)
        expected = _reference(QUERY, CANDIDATES, **kw)
        for g, e in zip(got, expected):
            assert (g is None) == (e is None)
            if e is not None:
                assert g == pytest.approx(e, abs=1e-9)

    def test_sequential_and_parallel_batch_kernels_agree(self):
        kw = dict(required_coverage=0.5, coverage_side="both")
        idx_seq = SentenceSimilarityIndex(CANDIDATES, parallel=False, **kw)
        idx_par = SentenceSimilarityIndex(CANDIDATES, parallel=True, **kw)
        seq = idx_seq.query(QUERY)
        par = idx_par.query(QUERY)
        for s, p in zip(seq, par):
            assert (s is None) == (p is None)
            if s is not None:
                assert s == pytest.approx(p, abs=1e-9)

    def test_empty_candidate_is_none_in_batch_path(self):
        idx = SentenceSimilarityIndex(CANDIDATES, parallel=True, required_coverage=0.5, coverage_side="max")
        got = idx.query(QUERY)
        assert got[5] is None  # CANDIDATES[5] == []

    def test_empty_query_short_circuits_before_kernel(self):
        idx = SentenceSimilarityIndex(CANDIDATES, parallel=True, required_coverage=0.5, coverage_side="max")
        assert idx.query([]) == [None] * len(CANDIDATES)


class TestBatchWithMatchesKernelScalesBeyondPurePythonLoop:
    """Not a formal perf assertion (machine-dependent), but pins that the coverage-gated batch path on a
    few thousand candidates completes in low single-digit seconds, not the multi-minute-plus wall clock the
    survey measured for the pure-Python per-candidate loop this replaces."""

    def test_few_thousand_candidates_parallel_completes_quickly(self):
        rng_words = ["ALPHA", "BETA", "GAMMA", "DELTA", "HEADACHE", "SEVERE", "PATIENT", "MILD", "PAIN", "NAUSEA"]
        candidates = [[rng_words[(i + j) % len(rng_words)] for j in range((i % 4) + 1)] for i in range(3000)]
        idx = SentenceSimilarityIndex(candidates, parallel=True, required_coverage=0.5, coverage_side="max")
        idx.query(QUERY)  # warm up numba JIT compilation, excluded from the timed call below
        start = time.perf_counter()
        idx.query(QUERY)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"coverage-gated batch query over 3000 candidates took {elapsed:.2f}s, expected well under 5s"
