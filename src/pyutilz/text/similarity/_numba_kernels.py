"""Numba ``@njit`` kernels backing the accelerated sentence-similarity paths.

Kept in ONE submodule on purpose: the kernels call each other directly (an njit
function referencing another njit function resolves it as a global at compile time),
and several carry MUST-STAY-IN-LOCKSTEP duplicated bodies whose divergence is only
reviewable side by side. ``HAS_NUMBA`` is defined here and re-exported downstream so
that a single import attempt decides which implementation set goes live.
"""

from __future__ import annotations

try:
    import numba as nb
    import numpy as np

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @nb.njit(cache=True)
    def _lev_dist_flat(buf, a_start, a_len, b_start, b_len):
        """Levenshtein distance between two words stored in a flat int32 buffer."""
        if a_len == 0:
            return b_len
        if b_len == 0:
            return a_len
        prev = np.empty(b_len + 1, dtype=np.int32)
        for j in range(b_len + 1):
            prev[j] = j
        for i in range(1, a_len + 1):
            curr_val = i
            ac = buf[a_start + i - 1]
            for j in range(1, b_len + 1):
                cost = 0 if ac == buf[b_start + j - 1] else 1
                ins = prev[j] + 1
                dele = curr_val + 1
                sub = prev[j - 1] + cost
                best = ins
                if dele < best:
                    best = dele
                if sub < best:
                    best = sub
                prev[j - 1] = curr_val
                curr_val = best
            prev[b_len] = curr_val
        return prev[b_len]

    @nb.njit(cache=True)
    def _sentences_similarity_core(buf, offsets, N_a, N_b, cMinLenTHreshold):
        """
        Core sentences_similarity in numba.

        buf: flat int32 array of all codepoints (words of A, then words of B)
        offsets: int32 array of length (N_a + N_b) — start index of each word in buf
                 Word i has codepoints buf[offsets[i]:offsets[i+1]] (offsets has N_a+N_b+1 entries)
        """
        if N_a < 1 or N_b < 1:
            return -1.0

        w_min = min(N_a, N_b)
        w_max = max(N_a, N_b)

        sim_res = np.zeros((N_a, N_b), dtype=np.float64)

        for i in range(N_a):
            a_start = offsets[i]
            cur_a_len = offsets[i + 1] - a_start
            if cur_a_len == 0:
                continue

            for j in range(N_b):
                bj = N_a + j
                b_start = offsets[bj]
                cur_b_len = offsets[bj + 1] - b_start
                if cur_b_len == 0:
                    continue

                t = max(cur_a_len, cur_b_len)
                lmin_len = min(cur_a_len, cur_b_len)

                # Exact match
                if cur_a_len == cur_b_len:
                    match = True
                    for c in range(cur_a_len):
                        if buf[a_start + c] != buf[b_start + c]:
                            match = False
                            break
                    if match:
                        sim_res[i, j] = 1.0
                        continue

                if lmin_len < cMinLenTHreshold:
                    continue

                # Prefix match. Requires a 2-character common prefix: a single shared character
                # otherwise earned the 0.9 floor, outscoring a correct suffix match.
                prefix_match = lmin_len >= 2
                for c in range(lmin_len):
                    if buf[a_start + c] != buf[b_start + c]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                # Full Levenshtein
                sim_min = 1.0 - _lev_dist_flat(buf, a_start, cur_a_len, b_start, cur_b_len) / t
                if cur_a_len == cur_b_len:
                    sim_res[i, j] = sim_min
                else:
                    # Sliding window
                    if cur_a_len < cur_b_len:
                        s_start, s_len = a_start, cur_a_len
                        l_start, l_len = b_start, cur_b_len
                    else:
                        s_start, s_len = b_start, cur_b_len
                        l_start, l_len = a_start, cur_a_len

                    best_gliding = s_len
                    for k in range(l_len - s_len + 1):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break

                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        # Greedy best-pair matching
        excluded_a = np.zeros(N_a, dtype=np.bool_)
        excluded_b = np.zeros(N_b, dtype=np.bool_)
        res = 0.0

        for _ in range(w_min):
            best_perf = 0.0
            best_i = 0
            best_j = 0
            for i in range(N_a):
                if excluded_a[i]:
                    continue
                for j in range(N_b):
                    if not excluded_b[j] and sim_res[i, j] >= best_perf:
                        best_perf = sim_res[i, j]
                        best_i = i
                        best_j = j
            res += sim_res[best_i, best_j]
            excluded_a[best_i] = True
            excluded_b[best_j] = True

        # Normalized by w_max, not w_min - see sentences_similarity()'s own comment on this same
        # line for why (short candidates otherwise out-score longer, more specific ones).
        if w_max > 0:
            res = res / w_max

        return res

    # Threshold (N_a*N_b product) above which the sort-once-then-scan greedy matcher
    # (_greedy_match_sorted, via _sentences_similarity_core_sorted) is used instead of the
    # plain O(w_min*N_a*N_b) rescan in _sentences_similarity_core. Full end-to-end benchmark
    # (matrix fill + greedy matching together, both paths JIT-warmed, best-of-15 perf_counter
    # reps, square N_a=N_b=N random-word matrices):
    #
    #   N      scan (ms)   sort (ms)   speedup (scan/sort)
    #     5       0.021       0.021          0.97x  (sort slower)
    #    10       0.078       0.080          0.97x  (sort slower)
    #    20       0.361       0.382          0.95x  (sort slower)
    #    50       2.767       2.898          0.95x  (sort slower)
    #   100      11.708      12.289          0.95x  (sort slower)
    #   200      46.597      56.756          0.82x  (sort slower)
    #   300      120.181     122.989          0.98x  (sort slower)
    #   500      358.745     365.884          0.98x  (~parity, noisy)
    #   550      550.133     516.094          1.07x  (sort faster)
    #   600      598.452     571.145          1.05x  (sort faster)
    #   650      682.300     636.695          1.07x  (sort faster)
    #   700      869.070     792.789          1.10x  (sort faster)
    #   800      994.418     894.832          1.11x  (sort faster)
    #  1000     1831.074    1510.506          1.21x  (sort faster)
    #
    # The sort variant loses (or is at best noise-level even) at every N up to ~500, and pulls
    # ahead from ~N=550 up -- np.argsort's fixed per-call overhead only pays for itself once the
    # O(N^2) matrix-fill + O(w*N^2) rescan cost is large enough. This is far above
    # SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD (50 tokens, N_a*N_b <= 2500 for the documented
    # use case), so the sort path essentially never fires for intended callers -- it only helps
    # the rare caller who deliberately exceeds the safety warning with a large near-square input.
    # Threshold set at N_a*N_b >= 550*550, safely past the noisy ~500 crossover zone measured above.
    #
    # A first attempt at _greedy_match_sorted quantized value+(i,j) into one int64 key (see
    # _greedy_match_sorted's docstring) and was WRONG: it silently collapsed real (non-tied)
    # float64 ULP-level differences into false ties, picking a different (and lower-scoring)
    # greedy match than _sentences_similarity_core on ~1.7% of random trials (8/480 in a
    # differential test across N=3..80, mixed and tie-heavy word sets). Fixed by sorting the
    # raw float64 bit pattern (exact, order-preserving for non-negative values) with a stable
    # sort instead of a lossy composite key -- see _greedy_match_sorted.
    _SORTED_MATCH_THRESHOLD = 550 * 550  # N_a*N_b product; see benchmark table above

    @nb.njit(cache=True)
    def _greedy_match_sorted(sim_res, N_a, N_b, w_min):
        """
        Sort-once-then-linear-scan greedy best-pair matcher.

        First attempt quantized value+(i,j) into a single int64 key (2**40 fractional bits for
        the value, low bits for i*N_b+j). That FAILED the differential test: two matrix cells can
        differ by a single float64 ULP (e.g. 0.33333333333333337 vs 0.3333333333333333, a ~5.5e-17
        gap from unrelated formulas both approximating 1/3) — real, not synthetic, since it showed
        up on random 5x8 word matrices. 2**40 quantization (~9e-13 granularity) collapses that into
        a false tie, so the (i, j) tie-break wins instead of the true (larger) value — wrong pick,
        and greedy matching cascades that into a materially different final score. Fixed by keeping
        the value comparison EXACT instead of quantized:

        sim_res is >= 0 everywhere (see _fill_sim_matrix), so reinterpreting each float64's raw
        bit pattern as int64 (`.view(np.int64)`) is order-preserving and lossless (IEEE-754: for
        non-negative floats, bit pattern compared as a non-negative integer matches float order
        exactly). Sorting those bit-pattern keys with a STABLE sort (kind='mergesort') ascending
        keeps exactly-tied cells in their original i-then-j enumeration order; reversing that order
        for the descending walk then visits the LAST-enumerated (largest i, then largest j) member
        of each tied group first — exactly the tie-break the original ascending `>=` rescan produces.
        No composite key, no quantization, no precision loss.
        """
        flat = sim_res.reshape(N_a * N_b)
        keys = flat.view(np.int64)
        order = np.argsort(keys, kind="mergesort")  # ascending, stable
        excluded_a = np.zeros(N_a, dtype=np.bool_)
        excluded_b = np.zeros(N_b, dtype=np.bool_)
        res = 0.0
        picked = 0
        for k in range(N_a * N_b - 1, -1, -1):  # walk descending
            if picked == w_min:
                break
            cell = order[k]
            i = cell // N_b
            j = cell % N_b
            if excluded_a[i] or excluded_b[j]:
                continue
            excluded_a[i] = True
            excluded_b[j] = True
            res += sim_res[i, j]
            picked += 1
        return res

    @nb.njit(cache=True)
    def _fill_sim_matrix(buf, offsets, N_a, N_b, cMinLenTHreshold):
        """Shared similarity-matrix fill, factored out of _sentences_similarity_core so both
        the plain-scan and sort-based greedy matchers can reuse it (used by _sentences_similarity_core_sorted)."""
        sim_res = np.zeros((N_a, N_b), dtype=np.float64)

        for i in range(N_a):
            a_start = offsets[i]
            cur_a_len = offsets[i + 1] - a_start
            if cur_a_len == 0:
                continue

            for j in range(N_b):
                bj = N_a + j
                b_start = offsets[bj]
                cur_b_len = offsets[bj + 1] - b_start
                if cur_b_len == 0:
                    continue

                t = max(cur_a_len, cur_b_len)
                lmin_len = min(cur_a_len, cur_b_len)

                # Exact match
                if cur_a_len == cur_b_len:
                    match = True
                    for c in range(cur_a_len):
                        if buf[a_start + c] != buf[b_start + c]:
                            match = False
                            break
                    if match:
                        sim_res[i, j] = 1.0
                        continue

                if lmin_len < cMinLenTHreshold:
                    continue

                # Prefix match. Requires a 2-character common prefix: a single shared character
                # otherwise earned the 0.9 floor, outscoring a correct suffix match.
                prefix_match = lmin_len >= 2
                for c in range(lmin_len):
                    if buf[a_start + c] != buf[b_start + c]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                # Full Levenshtein
                sim_min = 1.0 - _lev_dist_flat(buf, a_start, cur_a_len, b_start, cur_b_len) / t
                if cur_a_len == cur_b_len:
                    sim_res[i, j] = sim_min
                else:
                    # Sliding window
                    if cur_a_len < cur_b_len:
                        s_start, s_len = a_start, cur_a_len
                        l_start, l_len = b_start, cur_b_len
                    else:
                        s_start, s_len = b_start, cur_b_len
                        l_start, l_len = a_start, cur_a_len

                    best_gliding = s_len
                    for k in range(l_len - s_len + 1):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break

                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        return sim_res

    @nb.njit(cache=True)
    def _sentences_similarity_core_sorted(buf, offsets, N_a, N_b, cMinLenTHreshold):
        """Same result as _sentences_similarity_core, but the greedy best-pair matching pass
        uses _greedy_match_sorted (sort-once-then-scan) instead of the O(w_min*N_a*N_b) rescan.
        Only wins for large N — see _SORTED_MATCH_THRESHOLD and the benchmark note above it."""
        if N_a < 1 or N_b < 1:
            return -1.0

        w_min = min(N_a, N_b)
        w_max = max(N_a, N_b)

        sim_res = _fill_sim_matrix(buf, offsets, N_a, N_b, cMinLenTHreshold)
        res = _greedy_match_sorted(sim_res, N_a, N_b, w_min)

        # Normalized by w_max, not w_min - see sentences_similarity()'s own comment on this same
        # line for why (short candidates otherwise out-score longer, more specific ones).
        if w_max > 0:
            res = res / w_max

        return res

    @nb.njit(cache=True)
    def _greedy_match_with_tracking(sim_res, N_a, N_b, w_min):
        """Same greedy best-pair matching as the plain rescan in _sentences_similarity_core, but also
        returns per-word achieved match similarity for both sides (0.0 for a word left unmatched after
        w_min picks) - the raw material a coverage gate needs and the scalar-only core doesn't expose.
        Only used when a caller actually requests coverage; the default scalar-only cores are untouched."""
        excluded_a = np.zeros(N_a, dtype=np.bool_)
        excluded_b = np.zeros(N_b, dtype=np.bool_)
        matched_sim_a = np.zeros(N_a, dtype=np.float64)
        matched_sim_b = np.zeros(N_b, dtype=np.float64)
        res = 0.0
        for _ in range(w_min):
            best_perf = 0.0
            best_i = 0
            best_j = 0
            for i in range(N_a):
                if excluded_a[i]:
                    continue
                for j in range(N_b):
                    if not excluded_b[j] and sim_res[i, j] >= best_perf:
                        best_perf = sim_res[i, j]
                        best_i = i
                        best_j = j
            res += sim_res[best_i, best_j]
            matched_sim_a[best_i] = sim_res[best_i, best_j]
            matched_sim_b[best_j] = sim_res[best_i, best_j]
            excluded_a[best_i] = True
            excluded_b[best_j] = True
        return res, matched_sim_a, matched_sim_b

    @nb.njit(cache=True)
    def _sentences_similarity_core_with_matches(buf, offsets, N_a, N_b, cMinLenTHreshold):
        """Like _sentences_similarity_core, but also returns per-word matched similarities for both
        sides, for the coverage gate. Only called when a caller requests coverage - the plain scalar
        core stays the hot default path, unaffected."""
        w_min = min(N_a, N_b)
        w_max = max(N_a, N_b)
        sim_res = _fill_sim_matrix(buf, offsets, N_a, N_b, cMinLenTHreshold)
        res, matched_sim_a, matched_sim_b = _greedy_match_with_tracking(sim_res, N_a, N_b, w_min)
        if w_max > 0:
            res = res / w_max
        return res, matched_sim_a, matched_sim_b

    def _pack_words(words: list) -> tuple:
        """Pack a list of strings into (buf: int32[], offsets: int32[]) for numba.

        Uses numpy's frombuffer for fast ord() conversion instead of per-character Python loop.
        Returns (buf, offsets, n_words).

        - ``errors="surrogatepass"``: a lone/unpaired UTF-16 surrogate codepoint (producible via
          malformed-encoding recovery, surrogateescape, or hand-built strings) would otherwise
          raise UnicodeEncodeError on utf-32-le encoding -- unlike the pure-Python
          `sentences_similarity` reference, which never encodes anything and only crashes on such
          input if an actual jellyfish.levenshtein_distance comparison is reached. Passing
          surrogates through unchanged keeps this numba path from crashing on inputs the reference
          path tolerates via its exact-match fast path.
        - explicit little-endian dtype ("<u4", not a bare int32/native-order reinterpretation):
          the source bytes are produced as utf-32-**le** explicitly; reinterpreting them via a
          native-byte-order view (the previous ``array.array("i", ...)`` + int32-view approach)
          would silently byte-swap every codepoint above 0xFF on a big-endian platform.
        """
        n = len(words)
        parts = []
        offsets = np.empty(n + 1, dtype=np.int32)
        pos = 0
        for i, w in enumerate(words):
            offsets[i] = pos
            chunk = np.frombuffer(w.encode("utf-32-le", errors="surrogatepass"), dtype="<u4").astype(np.int32)
            parts.append(chunk)
            pos += len(chunk)
        offsets[n] = pos
        if pos > 0:
            buf = np.empty(pos, dtype=np.int32)
            p = 0
            for chunk in parts:
                buf[p : p + len(chunk)] = chunk
                p += len(chunk)
        else:
            buf = np.empty(0, dtype=np.int32)
        return buf, offsets, n

    def _run_sentences_similarity_core(buf: "np.ndarray", offsets: "np.ndarray", N_a: int, N_b: int, cMinLenTHreshold: int) -> float:
        """Dispatch to the sort-based greedy matcher above _SORTED_MATCH_THRESHOLD (N_a*N_b),
        else the plain rescan -- see the benchmark table above _SORTED_MATCH_THRESHOLD for why."""
        if N_a * N_b >= _SORTED_MATCH_THRESHOLD:
            return float(_sentences_similarity_core_sorted(buf, offsets, N_a, N_b, cMinLenTHreshold))
        return float(_sentences_similarity_core(buf, offsets, N_a, N_b, cMinLenTHreshold))

    @nb.njit(cache=True)
    def _compare_one_candidate(buf, offsets, query_n, cand_word_start, cn, cMinLenTHreshold):
        """Compare query (words 0..query_n-1) against one candidate starting at cand_word_start."""
        N_a = query_n
        N_b = cn
        w_min = min(N_a, N_b)
        w_max = max(N_a, N_b)

        sim_res = np.zeros((N_a, N_b), dtype=np.float64)

        for i in range(N_a):
            a_start = offsets[i]
            cur_a_len = offsets[i + 1] - a_start
            if cur_a_len == 0:
                continue
            for j in range(N_b):
                bj = cand_word_start + j
                b_start = offsets[bj]
                cur_b_len = offsets[bj + 1] - b_start
                if cur_b_len == 0:
                    continue

                t = max(cur_a_len, cur_b_len)
                lmin_len = min(cur_a_len, cur_b_len)

                if cur_a_len == cur_b_len:
                    match = True
                    for cc in range(cur_a_len):
                        if buf[a_start + cc] != buf[b_start + cc]:
                            match = False
                            break
                    if match:
                        sim_res[i, j] = 1.0
                        continue

                if lmin_len < cMinLenTHreshold:
                    continue

                # Prefix match. Requires a 2-character common prefix: a single shared character
                # otherwise earned the 0.9 floor, outscoring a correct suffix match.
                prefix_match = lmin_len >= 2
                for cc in range(lmin_len):
                    if buf[a_start + cc] != buf[b_start + cc]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                sim_min = 1.0 - _lev_dist_flat(buf, a_start, cur_a_len, b_start, cur_b_len) / t
                if cur_a_len == cur_b_len:
                    sim_res[i, j] = sim_min
                else:
                    if cur_a_len < cur_b_len:
                        s_start, s_len = a_start, cur_a_len
                        l_start, l_len = b_start, cur_b_len
                    else:
                        s_start, s_len = b_start, cur_b_len
                        l_start, l_len = a_start, cur_a_len

                    best_gliding = s_len
                    for k in range(l_len - s_len + 1):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break
                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        excluded_a = np.zeros(N_a, dtype=np.bool_)
        excluded_b = np.zeros(N_b, dtype=np.bool_)
        res = 0.0
        for _ in range(w_min):
            best_perf = 0.0
            best_i = 0
            best_j = 0
            for i in range(N_a):
                if excluded_a[i]:
                    continue
                for j in range(N_b):
                    if not excluded_b[j] and sim_res[i, j] >= best_perf:
                        best_perf = sim_res[i, j]
                        best_i = i
                        best_j = j
            res += sim_res[best_i, best_j]
            excluded_a[best_i] = True
            excluded_b[best_j] = True

        # Normalized by w_max, not w_min - see sentences_similarity()'s own comment on this same
        # line for why (short candidates otherwise out-score longer, more specific ones).
        if w_max > 0:
            res = res / w_max
        return res

    @nb.njit(cache=True)
    def _fill_sim_matrix_candidate(buf, offsets, query_n, cand_word_start, cn, cMinLenTHreshold):
        """Same matrix fill as `_compare_one_candidate`'s own inline loop, factored out so the
        coverage-tracking variant below can reuse it without duplicating/risking drift on the
        already-tested `_compare_one_candidate` itself."""
        N_a = query_n
        N_b = cn
        sim_res = np.zeros((N_a, N_b), dtype=np.float64)

        for i in range(N_a):
            a_start = offsets[i]
            cur_a_len = offsets[i + 1] - a_start
            if cur_a_len == 0:
                continue
            for j in range(N_b):
                bj = cand_word_start + j
                b_start = offsets[bj]
                cur_b_len = offsets[bj + 1] - b_start
                if cur_b_len == 0:
                    continue

                t = max(cur_a_len, cur_b_len)
                lmin_len = min(cur_a_len, cur_b_len)

                if cur_a_len == cur_b_len:
                    match = True
                    for cc in range(cur_a_len):
                        if buf[a_start + cc] != buf[b_start + cc]:
                            match = False
                            break
                    if match:
                        sim_res[i, j] = 1.0
                        continue

                if lmin_len < cMinLenTHreshold:
                    continue

                # Prefix match. Requires a 2-character common prefix: a single shared character
                # otherwise earned the 0.9 floor, outscoring a correct suffix match.
                prefix_match = lmin_len >= 2
                for cc in range(lmin_len):
                    if buf[a_start + cc] != buf[b_start + cc]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                sim_min = 1.0 - _lev_dist_flat(buf, a_start, cur_a_len, b_start, cur_b_len) / t
                if cur_a_len == cur_b_len:
                    sim_res[i, j] = sim_min
                else:
                    if cur_a_len < cur_b_len:
                        s_start, s_len = a_start, cur_a_len
                        l_start, l_len = b_start, cur_b_len
                    else:
                        s_start, s_len = b_start, cur_b_len
                        l_start, l_len = a_start, cur_a_len

                    best_gliding = s_len
                    for k in range(l_len - s_len + 1):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break
                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        return sim_res

    @nb.njit(cache=True)
    def _compare_one_candidate_with_matches(buf, offsets, query_n, cand_word_start, cn, cMinLenTHreshold):
        """Like `_compare_one_candidate`, but also returns per-word matched similarities for both
        sides, for `SentenceSimilarityIndex`'s coverage gate. Only called when coverage is requested."""
        N_a = query_n
        N_b = cn
        w_min = min(N_a, N_b)
        w_max = max(N_a, N_b)
        sim_res = _fill_sim_matrix_candidate(buf, offsets, query_n, cand_word_start, cn, cMinLenTHreshold)
        res, matched_sim_a, matched_sim_b = _greedy_match_with_tracking(sim_res, N_a, N_b, w_min)
        if w_max > 0:
            res = res / w_max
        return res, matched_sim_a, matched_sim_b

    @nb.njit(cache=True)
    def _sentences_similarity_batch_core(buf, offsets, word_counts, n_candidates, query_n: int, cMinLenTHreshold):
        """Compare one query against multiple candidates sequentially.

        MUST STAY IN LOCKSTEP with `_sentences_similarity_batch_parallel`: the per-candidate body is duplicated,
        and only the loop header (`range` vs `nb.prange`) and where `cand_starts` comes from
        (computed here vs supplied by the caller) may differ. The two are separate functions, not one factory-generated body,
        because numba's `cache=True` cannot cache a closure-generated function -- losing the on-disk
        cache costs a multi-second re-JIT on every fresh process. The pair is selected by a
        candidate-count threshold, so a fix applied to one and not the other makes the SAME input
        score differently depending only on batch size. `test_similarity_kernel_twin_parity.py`
        pins the two against each other.
        """
        results = np.empty(n_candidates, dtype=np.float64)
        # Pre-compute cumulative word starts for each candidate
        cand_starts = np.empty(n_candidates, dtype=np.int32)
        pos = query_n
        for c in range(n_candidates):
            cand_starts[c] = pos
            pos += word_counts[c + 1]

        for c in range(n_candidates):
            cn = word_counts[c + 1]
            if cn < 1:
                results[c] = -1.0
            else:
                results[c] = _compare_one_candidate(buf, offsets, query_n, cand_starts[c], cn, cMinLenTHreshold)
        return results

    @nb.njit(parallel=True, cache=True)
    def _sentences_similarity_batch_parallel(buf, offsets, word_counts, cand_starts, n_candidates, query_n, cMinLenTHreshold):
        """Compare one query against multiple candidates in parallel threads.

        MUST STAY IN LOCKSTEP with `_sentences_similarity_batch_core`: the per-candidate body is duplicated,
        and only the loop header (`range` vs `nb.prange`) and where `cand_starts` comes from
        (computed here vs supplied by the caller) may differ. The two are separate functions, not one factory-generated body,
        because numba's `cache=True` cannot cache a closure-generated function -- losing the on-disk
        cache costs a multi-second re-JIT on every fresh process. The pair is selected by a
        candidate-count threshold, so a fix applied to one and not the other makes the SAME input
        score differently depending only on batch size. `test_similarity_kernel_twin_parity.py`
        pins the two against each other.
        """
        results = np.empty(n_candidates, dtype=np.float64)
        for c in nb.prange(n_candidates):  # type: ignore[attr-defined]
            cn = word_counts[c + 1]
            if cn < 1:
                results[c] = -1.0
            else:
                results[c] = _compare_one_candidate(buf, offsets, query_n, cand_starts[c], cn, cMinLenTHreshold)
        return results

    @nb.njit(cache=True)
    def _sentences_similarity_batch_with_matches_core(buf, offsets, word_counts, cand_starts, n_candidates, query_n, cMinLenTHreshold):
        """Like `_sentences_similarity_batch_core`, but calls `_compare_one_candidate_with_matches` instead
        of the scalar-only `_compare_one_candidate`, exposing per-word match tracking for the coverage gate.
        Returns (results, matched_sim_a_all, matched_sim_b_flat): `matched_sim_a_all` is
        `(n_candidates, query_n)` since every candidate compares against the same query length;
        `matched_sim_b_flat` uses the same flat per-candidate-word layout as `buf`/`offsets` (total
        candidate words long, one contiguous run per candidate at `cand_starts[c] - query_n`), since each
        candidate's own word count varies and can't share a single 2-D array shape.

        MUST STAY BYTE-IDENTICAL to `_sentences_similarity_batch_with_matches_parallel` except for the loop header
        (`range` vs `nb.prange`). The two are separate functions, not one factory-generated body,
        because numba's `cache=True` cannot cache a closure-generated function -- losing the on-disk
        cache costs a multi-second re-JIT on every fresh process. The pair is selected by a
        candidate-count threshold, so a fix applied to one and not the other makes the SAME input
        score differently depending only on batch size. `test_similarity_kernel_twin_parity.py`
        pins the two against each other.
        """
        results = np.empty(n_candidates, dtype=np.float64)
        matched_sim_a_all = np.zeros((n_candidates, query_n), dtype=np.float64)
        total_cand_words = 0
        for c in range(n_candidates):
            total_cand_words += word_counts[c + 1]
        matched_sim_b_flat = np.zeros(total_cand_words, dtype=np.float64)
        for c in range(n_candidates):
            cn = word_counts[c + 1]
            if cn < 1:
                results[c] = -1.0
                continue
            res, matched_a, matched_b = _compare_one_candidate_with_matches(buf, offsets, query_n, cand_starts[c], cn, cMinLenTHreshold)
            results[c] = res
            matched_sim_a_all[c, :] = matched_a
            b_start = cand_starts[c] - query_n
            for k in range(cn):
                matched_sim_b_flat[b_start + k] = matched_b[k]
        return results, matched_sim_a_all, matched_sim_b_flat

    @nb.njit(parallel=True, cache=True)
    def _sentences_similarity_batch_with_matches_parallel(buf, offsets, word_counts, cand_starts, n_candidates, query_n, cMinLenTHreshold):
        """`nb.prange`-parallel twin of `_sentences_similarity_batch_with_matches_core` - the batched
        "with-matches" kernel the coverage gate needs at scale (`SentenceSimilarityIndex.query`'s
        coverage-active path previously fell back to a pure-Python per-candidate loop calling the same
        `_compare_one_candidate_with_matches`, which is what made the SNOMED bridge (383,345 candidates)
        and complaint-parser retrieval (11,555 candidates) infeasible - see this repo's own commit history
        and autopsia's `audits/2026-08-13-similarity-coverage-gate-survey/survey.md`). Same flat-layout
        return convention as the sequential core above; each `c` writes only its own row/slice, so no
        cross-thread aliasing despite the shared output arrays.

        MUST STAY BYTE-IDENTICAL to `_sentences_similarity_batch_with_matches_core` except for the loop header
        (`range` vs `nb.prange`). The two are separate functions, not one factory-generated body,
        because numba's `cache=True` cannot cache a closure-generated function -- losing the on-disk
        cache costs a multi-second re-JIT on every fresh process. The pair is selected by a
        candidate-count threshold, so a fix applied to one and not the other makes the SAME input
        score differently depending only on batch size. `test_similarity_kernel_twin_parity.py`
        pins the two against each other.
        """
        results = np.empty(n_candidates, dtype=np.float64)
        matched_sim_a_all = np.zeros((n_candidates, query_n), dtype=np.float64)
        total_cand_words = 0
        for c in range(n_candidates):
            total_cand_words += word_counts[c + 1]
        matched_sim_b_flat = np.zeros(total_cand_words, dtype=np.float64)
        for c in nb.prange(n_candidates):  # type: ignore[attr-defined]
            cn = word_counts[c + 1]
            if cn < 1:
                results[c] = -1.0
                continue
            res, matched_a, matched_b = _compare_one_candidate_with_matches(buf, offsets, query_n, cand_starts[c], cn, cMinLenTHreshold)
            results[c] = res
            matched_sim_a_all[c, :] = matched_a
            b_start = cand_starts[c] - query_n
            for k in range(cn):
                matched_sim_b_flat[b_start + k] = matched_b[k]
        return results, matched_sim_a_all, matched_sim_b_flat
