"""Parity between the sequential and `nb.prange`-parallel twins of the batched similarity kernels.

The four kernels are two duplicated bodies (a `range` copy and an `nb.prange` copy) kept apart only
because numba's `cache=True` cannot cache a factory-generated closure. Nothing but these tests stops a
fix landing in one copy and not the other -- and because the pair is chosen by a candidate-count
threshold, that divergence would make the SAME input score differently depending on batch size.
"""

import random

import numpy as np
import pytest

similarity = pytest.importorskip("pyutilz.text.similarity")

if not getattr(similarity, "HAS_NUMBA", False):
    pytest.skip("numba not installed", allow_module_level=True)

from pyutilz.text.similarity import _numba_api, _numba_kernels  # noqa: E402


def _random_batch(seed: int, n_candidates: int):
    rnd = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "et", "theta"]

    def words(n):
        return [rnd.choice(vocab) for _ in range(n)]

    query = words(rnd.randint(1, 5))
    candidates = [words(rnd.randint(1, 6)) for _ in range(n_candidates)]
    # A degenerate (empty) candidate exercises the `if cn < 1: results[c] = -1.0` convention, which is
    # exactly the kind of edge case a one-sided fix would diverge on.
    candidates[0] = []
    return query, candidates


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_batch_core_and_parallel_agree(seed):
    query, candidates = _random_batch(seed, 9)
    buf, offsets, wc, cand_starts, n_query = _numba_api._prepare_batch(query, candidates)

    seq = _numba_kernels._sentences_similarity_batch_core(buf, offsets, wc, len(candidates), n_query, 1)
    par = _numba_kernels._sentences_similarity_batch_parallel(buf, offsets, wc, cand_starts, len(candidates), n_query, 1)

    np.testing.assert_allclose(np.asarray(seq), np.asarray(par), rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_batch_with_matches_core_and_parallel_agree(seed):
    query, candidates = _random_batch(seed, 9)
    buf, offsets, wc, cand_starts, n_query = _numba_api._prepare_batch(query, candidates)

    seq = _numba_kernels._sentences_similarity_batch_with_matches_core(buf, offsets, wc, cand_starts, len(candidates), n_query, 1)
    par = _numba_kernels._sentences_similarity_batch_with_matches_parallel(buf, offsets, wc, cand_starts, len(candidates), n_query, 1)

    for got, expected in zip(par, seq):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0, atol=0)


def test_degenerate_candidate_convention_is_shared():
    """Both twins must mark an empty candidate with the -1.0 sentinel, not 0.0 or NaN."""
    query, candidates = _random_batch(3, 4)
    buf, offsets, wc, cand_starts, n_query = _numba_api._prepare_batch(query, candidates)

    seq = _numba_kernels._sentences_similarity_batch_core(buf, offsets, wc, len(candidates), n_query, 1)
    par = _numba_kernels._sentences_similarity_batch_parallel(buf, offsets, wc, cand_starts, len(candidates), n_query, 1)
    assert seq[0] == -1.0
    assert par[0] == -1.0
