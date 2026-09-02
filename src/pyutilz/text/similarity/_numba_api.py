"""Python-level API over the numba kernels, plus the pure-Python fallbacks used when
numba is not installed.

Split out of the historical flat ``pyutilz.text.similarity`` module; re-exported
from the package ``__init__`` to preserve the public import surface. Exactly one of
the two branches below is ever live in a process, mirroring the original
``try: import numba ... except ImportError:`` structure.
"""

from __future__ import annotations

from ._common import Optional, _validate_coverage_side
from ._numba_kernels import HAS_NUMBA
from .sentences import sentences_similarity

if HAS_NUMBA:
    import numpy as np

    from ._common import SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD, _check_word_coverage, _strip_stop_words, logger
    from ._numba_kernels import (
        _compare_one_candidate_with_matches,  # noqa: F401 - re-exported for the historical pyutilz.text.similarity.<name> surface
        _pack_words,
        _run_sentences_similarity_core,
        _sentences_similarity_batch_core,
        _sentences_similarity_batch_parallel,
        _sentences_similarity_batch_with_matches_core,
        _sentences_similarity_batch_with_matches_parallel,
        _sentences_similarity_core_with_matches,
    )

    def pack_sentence(words: list) -> Optional[tuple]:
        """Pre-pack a word list for repeated use with sentences_similarity_numba.

        Returns an opaque tuple that can be passed to sentences_similarity_numba_packed().
        Useful when comparing one query against many candidates — pack candidates once.

            >>> packed = pack_sentence(["HELLO", "WORLD"])
            >>> packed is not None
            True
        """
        if not words:
            return None
        buf, offsets, n = _pack_words(words)
        return (buf, offsets, n)

    def sentences_similarity_numba(
        SentenceA: list,
        SentenceB: list,
        cMinLenTHreshold: int = 1,
        min_word_similarity: float = 0.5,
        required_coverage: Optional[float] = None,
        required_matched_words: Optional[int] = None,
        coverage_side: str = "max",
        stop_words: Optional[list] = None,
    ) -> Optional[float]:
        """
        Numba-accelerated version of sentences_similarity.

        Same algorithm, but Levenshtein + sliding window + greedy matching all run in compiled code.
        First call triggers JIT compilation (~1-2s), subsequent calls are 5-20x faster.

            >>> sentences_similarity_numba(["HELLO", "WORLD"], ["HELLO", "WORLD"])
            1.0

            >>> sentences_similarity_numba([], ["TEST"])

        Same O(w * N^2) greedy-matching complexity as sentences_similarity (compiled, so much
        faster in absolute terms, but the same asymptotic shape) — a warning is logged past
        SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD (50) tokens.

        Same optional coverage gate as `sentences_similarity` (see its docstring) - `required_coverage`/
        `required_matched_words` opt in, off by default, zero overhead on the default path (the plain
        scalar-only core is still used when neither is set; only the coverage-active path recomputes via
        `_sentences_similarity_core_with_matches`, which also tracks per-word matches).
        """
        SentenceA = _strip_stop_words(SentenceA, stop_words)
        SentenceB = _strip_stop_words(SentenceB, stop_words)
        N_a = len(SentenceA)
        N_b = len(SentenceB)
        if N_a < 1 or N_b < 1:
            return None
        if N_a > SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD or N_b > SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD:
            logger.warning(
                "sentences_similarity_numba called with N_a=%d, N_b=%d tokens, exceeding the safe threshold of %d. "
                "The greedy matching pass is O(w_min * N_a * N_b); expect quadratic-ish slowdown on long inputs.",
                N_a,
                N_b,
                SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD,
            )

        coverage_active = required_coverage is not None or required_matched_words is not None
        if coverage_active:
            _validate_coverage_side(coverage_side)

        # Fast packing via utf-32-le bulk conversion
        all_words = SentenceA + SentenceB
        buf, offsets, _ = _pack_words(all_words)

        if coverage_active:
            result, matched_sim_a, matched_sim_b = _sentences_similarity_core_with_matches(buf, offsets, N_a, N_b, cMinLenTHreshold)
            if result < 0:
                return None
            stop_set = set(stop_words) if stop_words else set()
            ok = True
            if coverage_side in ("max", "both"):
                words, matched = (SentenceA, matched_sim_a) if N_a >= N_b else (SentenceB, matched_sim_b)
                ok = ok and _check_word_coverage(words, matched, min_word_similarity, required_coverage, required_matched_words, stop_set)
            if ok and coverage_side in ("min", "both"):
                words, matched = (SentenceB, matched_sim_b) if N_a >= N_b else (SentenceA, matched_sim_a)
                ok = ok and _check_word_coverage(words, matched, min_word_similarity, required_coverage, required_matched_words, stop_set)
            if not ok:
                return None
            return float(result)

        result = _run_sentences_similarity_core(buf, offsets, N_a, N_b, cMinLenTHreshold)
        if result < 0:
            return None
        # float(...) keeps the return type identical whether numba actually JIT-compiled this
        # (native Python float, auto-unboxed) or NUMBA_DISABLE_JIT=1 is set (plain numpy call,
        # which would otherwise leak a np.float64 scalar and break reprs/doctests downstream).
        return float(result)

    def sentences_similarity_numba_packed(packed_a: Optional[tuple], packed_b: Optional[tuple], cMinLenTHreshold: int = 1) -> Optional[float]:
        """
        Numba-accelerated sentences_similarity with pre-packed inputs.

        Use pack_sentence() to pre-pack word lists, then call this for each comparison.
        Eliminates Python→numpy packing overhead when comparing one query against many candidates.

            >>> a = pack_sentence(["HELLO", "WORLD"])
            >>> b = pack_sentence(["HELLO", "WORLD"])
            >>> sentences_similarity_numba_packed(a, b)
            1.0
        """
        if packed_a is None or packed_b is None:
            return None
        buf_a, off_a, n_a = packed_a
        buf_b, off_b, n_b = packed_b

        # Concatenate buffers and adjust offsets
        buf = np.concatenate((buf_a, buf_b))
        total_a_chars = off_a[n_a]
        offsets = np.concatenate((off_a[:n_a], off_b[:n_b] + total_a_chars, np.array([off_b[n_b] + total_a_chars], dtype=np.int32)))

        result = _run_sentences_similarity_core(buf, offsets, n_a, n_b, cMinLenTHreshold)
        if result < 0:
            return None
        # float(...): see sentences_similarity_numba's identical comment -- keeps the return
        # type stable regardless of whether numba actually JIT-compiled this call.
        return float(result)

    def _prepare_batch(query_words, candidates):
        """Pack query + candidates into flat buffers. Returns (buf, offsets, wc, cand_starts, n_query)."""
        n_query = len(query_words)
        all_words = list(query_words)
        word_counts = [n_query]
        for cand in candidates:
            word_counts.append(len(cand))
            all_words.extend(cand)
        buf, offsets, _ = _pack_words(all_words)
        wc = np.array(word_counts, dtype=np.int32)
        # Pre-compute candidate word start indices
        cand_starts = np.empty(len(candidates), dtype=np.int32)
        pos = n_query
        for i in range(len(candidates)):
            cand_starts[i] = pos
            pos += word_counts[i + 1]
        return buf, offsets, wc, cand_starts, n_query

    def sentences_similarity_numba_batch(query_words: list, candidates: list, cMinLenTHreshold: int = 1, parallel: bool = False) -> list:
        """
        Compare one query against many candidates in a single numba call.

        Eliminates per-candidate Python→numba overhead entirely.
        Set parallel=True to use multiple CPU threads via numba.prange.
        Returns list of float similarities (None for empty candidates).

            >>> sentences_similarity_numba_batch(["AAA"], [["AAA"], ["ZZZ"]])
            [1.0, 0.0]
        """
        n_query = len(query_words)
        if n_query < 1:
            return [None] * len(candidates)

        buf, offsets, wc, cand_starts, n_query = _prepare_batch(query_words, candidates)

        if parallel:
            raw = _sentences_similarity_batch_parallel(buf, offsets, wc, cand_starts, len(candidates), n_query, cMinLenTHreshold)
        else:
            raw = _sentences_similarity_batch_core(buf, offsets, wc, len(candidates), n_query, cMinLenTHreshold)
        return [None if v < 0 else float(v) for v in raw]

    class SentenceSimilarityIndex:
        """Pre-built index of candidates for fast repeated queries.

        Build once with a list of candidate token lists, then call query() for each
        new input. All candidate tokenization and numpy packing happens at build time.
        Set parallel=True at construction to use multiple CPU threads.

            >>> idx = SentenceSimilarityIndex([["AAA"], ["ZZZ"]])
            >>> idx.query(["AAA"])
            [1.0, 0.0]
        """

        def __init__(
            self,
            candidates: list[list[str]],
            cMinLenTHreshold: int = 1,
            parallel: bool = False,
            min_word_similarity: float = 0.5,
            required_coverage: Optional[float] = None,
            required_matched_words: Optional[int] = None,
            coverage_side: str = "max",
            stop_words: Optional[list] = None,
        ):
            self.cMinLenTHreshold = cMinLenTHreshold
            self.parallel = parallel
            self.n_candidates = len(candidates)
            self.min_word_similarity = min_word_similarity
            self.required_coverage = required_coverage
            self.required_matched_words = required_matched_words
            self.coverage_side = coverage_side
            self.stop_words = stop_words
            self._coverage_active = required_coverage is not None or required_matched_words is not None
            if self._coverage_active:
                _validate_coverage_side(coverage_side)
            # Stop words stripped BEFORE packing (2026-08-16, see `_strip_stop_words`'s own docstring for
            # why the base w_max-normalized score, not just the opt-in coverage gate, needs this) - kept
            # filtered here too (not just in the packed buffer) so the coverage gate's own per-word
            # `matched_sim_b` array (built against the packed, filtered length) stays index-aligned with
            # `self._candidates[idx]` below.
            candidates = [_strip_stop_words(c, stop_words) for c in candidates]
            self._candidates = candidates  # kept for the coverage gate's word-list lookups
            # Pre-pack all candidates into a single flat buffer
            all_cand_words = []
            self._word_counts = []  # word count per candidate
            for cand in candidates:
                self._word_counts.append(len(cand))
                all_cand_words.extend(cand)
            self._cand_buf, self._cand_offsets, _ = _pack_words(all_cand_words)
            self._wc_arr = np.array(self._word_counts, dtype=np.int32)
            # Pre-compute candidate word start indices
            self._cand_starts = np.empty(self.n_candidates, dtype=np.int32)
            pos = 0
            for i in range(self.n_candidates):
                self._cand_starts[i] = pos
                pos += self._word_counts[i]

        def query(self, query_words: list[str]) -> list:
            """Compare query against all indexed candidates. Returns list of similarities."""
            query_words = _strip_stop_words(query_words, self.stop_words)
            n_query = len(query_words)
            if n_query < 1:
                return [None] * self.n_candidates

            # Pack only the query (candidates are pre-packed)
            q_buf, q_offsets, _ = _pack_words(query_words)

            # Concatenate query + pre-packed candidates
            buf = np.concatenate((q_buf, self._cand_buf))
            total_q_chars = q_offsets[n_query]
            # Build combined offsets: query offsets + shifted candidate offsets
            shifted_cand = self._cand_offsets[:-1] + total_q_chars
            offsets = np.concatenate((
                q_offsets[:n_query],
                shifted_cand,
                np.array([self._cand_offsets[-1] + total_q_chars], dtype=np.int32),
            ))
            wc = np.concatenate((np.array([n_query], dtype=np.int32), self._wc_arr))
            # Shift cand_starts by n_query (query words go first in offsets)
            cand_starts = self._cand_starts + n_query

            if not self._coverage_active:
                if self.parallel:
                    raw = _sentences_similarity_batch_parallel(buf, offsets, wc, cand_starts, self.n_candidates, n_query, self.cMinLenTHreshold)
                else:
                    raw = _sentences_similarity_batch_core(buf, offsets, wc, self.n_candidates, n_query, self.cMinLenTHreshold)
                return [None if v < 0 else float(v) for v in raw]

            # Coverage-active path: needs per-candidate word-level match tracking. Uses the batched
            # "with-matches" kernels (`_sentences_similarity_batch_with_matches_parallel`/`_core`), which
            # call `_compare_one_candidate_with_matches` inside a single numba call (prange-parallel when
            # self.parallel) instead of looping per-candidate in pure Python - the latter was measured
            # infeasible at scale (383,345/11,555-candidate pools; see autopsia's
            # audits/2026-08-13-similarity-coverage-gate-survey/survey.md).
            stop_set = set(self.stop_words) if self.stop_words else set()
            if self.parallel:
                raw, matched_sim_a_all, matched_sim_b_flat = _sentences_similarity_batch_with_matches_parallel(
                    buf, offsets, wc, cand_starts, self.n_candidates, n_query, self.cMinLenTHreshold
                )
            else:
                raw, matched_sim_a_all, matched_sim_b_flat = _sentences_similarity_batch_with_matches_core(
                    buf, offsets, wc, cand_starts, self.n_candidates, n_query, self.cMinLenTHreshold
                )
            results: list = []
            b_flat_pos = 0
            for idx in range(self.n_candidates):
                cn = int(self._wc_arr[idx])
                if cn < 1:
                    results.append(None)
                    continue
                res = raw[idx]
                matched_sim_a = matched_sim_a_all[idx]
                matched_sim_b = matched_sim_b_flat[b_flat_pos : b_flat_pos + cn]
                b_flat_pos += cn
                cand_words = self._candidates[idx]
                ok = True
                if self.coverage_side in ("max", "both"):
                    words, matched = (query_words, matched_sim_a) if n_query >= cn else (cand_words, matched_sim_b)
                    ok = ok and _check_word_coverage(words, matched, self.min_word_similarity, self.required_coverage, self.required_matched_words, stop_set)
                if ok and self.coverage_side in ("min", "both"):
                    words, matched = (cand_words, matched_sim_b) if n_query >= cn else (query_words, matched_sim_a)
                    ok = ok and _check_word_coverage(words, matched, self.min_word_similarity, self.required_coverage, self.required_matched_words, stop_set)
                results.append(float(res) if ok else None)
            return results

else:
    def sentences_similarity_numba(
        SentenceA: list,
        SentenceB: list,
        cMinLenTHreshold: int = 1,
        min_word_similarity: float = 0.5,
        required_coverage: Optional[float] = None,
        required_matched_words: Optional[int] = None,
        coverage_side: str = "max",
        stop_words: Optional[list] = None,
    ) -> Optional[float]:
        """Fallback to pure-Python version when numba is not installed."""
        return sentences_similarity(
            SentenceA,
            SentenceB,
            cMinLenTHreshold,
            min_word_similarity=min_word_similarity,
            required_coverage=required_coverage,
            required_matched_words=required_matched_words,
            coverage_side=coverage_side,
            stop_words=stop_words,
        )

    def sentences_similarity_numba_packed(packed_a: Optional[tuple], packed_b: Optional[tuple], cMinLenTHreshold: int = 1) -> Optional[float]:
        """Fallback to pure-Python version when numba is not installed."""
        return sentences_similarity(list(packed_a), list(packed_b), cMinLenTHreshold) if packed_a and packed_b else None

    def sentences_similarity_numba_batch(query_words: list, candidates: list, cMinLenTHreshold: int = 1, parallel: bool = False) -> list:
        """Fallback to pure-Python version when numba is not installed."""
        return [sentences_similarity(query_words, c, cMinLenTHreshold) for c in candidates]

    def pack_sentence(words: list) -> Optional[tuple]:
        """Fallback — returns words as-is."""
        return tuple(words) if words else None

    class SentenceSimilarityIndex:  # type: ignore[no-redef]  # pure-Python fallback of the numba-accelerated class above; only one definition is ever live per process
        """Fallback without numba — uses pure Python."""
        def __init__(
            self,
            candidates: list,
            cMinLenTHreshold: int = 1,
            parallel: bool = False,
            min_word_similarity: float = 0.5,
            required_coverage: Optional[float] = None,
            required_matched_words: Optional[int] = None,
            coverage_side: str = "max",
            stop_words: Optional[list] = None,
        ):
            self.candidates = candidates
            self.cMinLenTHreshold = cMinLenTHreshold
            self.n_candidates = len(candidates)
            self.min_word_similarity = min_word_similarity
            self.required_coverage = required_coverage
            self.required_matched_words = required_matched_words
            self.coverage_side = coverage_side
            self.stop_words = stop_words
            if required_coverage is not None or required_matched_words is not None:
                _validate_coverage_side(coverage_side)
        def query(self, query_words: list) -> list:
            """Compare query against all indexed candidates. Returns list of similarities."""
            return [
                sentences_similarity(
                    query_words,
                    c,
                    self.cMinLenTHreshold,
                    min_word_similarity=self.min_word_similarity,
                    required_coverage=self.required_coverage,
                    required_matched_words=self.required_matched_words,
                    coverage_side=self.coverage_side,
                    stop_words=self.stop_words,
                )
                for c in self.candidates
            ]
