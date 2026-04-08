# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any

from jellyfish import levenshtein_distance
from pyutilz.text.strings import strip_doubled_characters


def levenshtein_strings_similarity(a: str, b: str) -> float:
    """
        simple string percent similarity

        >>> levenshtein_strings_similarity("MeasureOIS21", "MeasureOIS18")
        0.8333333333333334

    """
    return 1 - levenshtein_distance(a, b) / max(len(a), len(b))


def contigous_strings_similarity(a: str, b: str) -> tuple:
    """
        similarity of 2 strings measured in contigous blocks either from the left, from the right, or both (non-symmetrically).
        it's more strong than just levenshtein_distance.

        >>> contigous_strings_similarity("MeosureOIS21qwe", "MeasureOIS18qwe")
        (0.3333333333333333, 'Meqwe')

        >>> contigous_strings_similarity("MeosureOIS21qwe", "MeosureOIS21qwe")
        (1.0, 'MeosureOIS21qwe')
    """
    best_l, best_r, best_m = 0, 0, 0

    min_length = min(len(a), len(b))
    for i in range(1, min_length + 1):
        if a[:i] == b[:i]:
            best_l = i
        if a[-i:] == b[-i:]:
            best_r = i

    best_m = best_l + best_r

    if best_l + best_r >= min_length:
        root = a
        best_m = min_length
    else:
        root = a[:best_l]
        if best_r > 0:
            root += a[-best_r:]

    return max(best_l, best_r, best_m) / max(len(a), len(b)), root


def sentences_similarity(SentenceA: list, SentenceB: list, cMinLenTHreshold: int = 1) -> float:
    """
    Улучшенный алгоритм распознавания одинаковых фраз, в т.ч. спортивных команд/игроков, адресов.

    Альтернатива пакетам difflib (stdlib SequenceMatcher), rapidfuzz, jellyfish (Levenshtein),
    а также TF-IDF + cosine similarity из sklearn. На бенчмарке fuzzy-матчинга коротких строк
    показывает то же качество (73.3%), что и лучший из них (keyword_overlap / token Jaccard),
    при этом имеет Numba-ускоренные варианты (sentences_similarity_numba, SentenceSimilarityIndex).

    Для каждой пары слов из А,В рассчитывам максимальную и минимальную похожести.
    Минимальная - это просто метрическое расстояние.
    При неодинаковых длинах слов максимальная похожесть отличается от минимальной,
    и представляет собой лучший результат сравнения при скольжении меньшего слова вдоль большего.

        >>> sentences_similarity(["HELLO", "WORLD"], ["HELLO", "WORLD"])
        1.0

        >>> sentences_similarity([], ["TEST"])

    """
    N_a = len(SentenceA)
    N_b = len(SentenceB)
    if N_a < 1 or N_b < 1:
        return
    if N_a < N_b:
        w_min = N_a
        w_max = N_b
    else:
        w_min = N_b
        w_max = N_a

    # Plain lists instead of numpy for typical small inputs (3-10 words)
    Sim_res = [[0.0] * N_b for _ in range(N_a)]

    for i in range(N_a):
        a = SentenceA[i]
        cur_a_len = len(a)
        if cur_a_len == 0:
            continue
        for j in range(N_b):
            b = SentenceB[j]
            cur_b_len = len(b)
            if cur_b_len == 0:
                continue

            t = max(cur_a_len, cur_b_len)
            lminLen = min(cur_a_len, cur_b_len)

            # Exact match — early exit (most common case for matching words)
            if a == b:
                Sim_res[i][j] = 1.0
                continue

            # Check if starts match (one is prefix of another)
            if a[:lminLen] == b[:lminLen]:
                # Чтобы "Almeria B"<>"Al-Budaiya"
                sim = 0.9 + 0.1 * lminLen / t
                Sim_res[i][j] = sim
                continue

            if lminLen < cMinLenTHreshold:
                continue

            # Levenshtein distance
            sim_min = 1 - levenshtein_distance(a, b) / t
            if cur_a_len == cur_b_len:
                Sim_res[i][j] = sim_min
            else:
                # Скольжение меньшего слова вдоль большего
                if cur_a_len < cur_b_len:
                    short, long, short_len, long_len = a, b, cur_a_len, cur_b_len
                else:
                    short, long, short_len, long_len = b, a, cur_b_len, cur_a_len

                best_gliding_perf = short_len  # worst case = all chars different
                for k in range(long_len - short_len):
                    t_sim = levenshtein_distance(short, long[k : k + short_len])
                    if t_sim < best_gliding_perf:
                        best_gliding_perf = t_sim
                        if t_sim == 0:
                            break  # perfect substring match

                sim_max = 1 - best_gliding_perf / short_len
                Sim_res[i][j] = 0.5 * (sim_max + sim_min)

    # Greedy best-pair matching: pick highest similarity pair, exclude both words, repeat
    excluded_a = [False] * N_a
    excluded_b = [False] * N_b
    res = 0.0

    for _ in range(w_min):
        best_perf = 0.0
        best_i = 0
        best_j = 0
        for i in range(N_a):
            if excluded_a[i]:
                continue
            row = Sim_res[i]
            for j in range(N_b):
                if not excluded_b[j] and row[j] >= best_perf:
                    best_perf = row[j]
                    best_i = i
                    best_j = j

        res += Sim_res[best_i][best_j]
        excluded_a[best_i] = True
        excluded_b[best_j] = True

    if w_min > 0:
        res = res / w_min * (1 - (w_max - w_min) / (w_max + w_min) / 5)
    else:
        res = 0

    return res


def normalize_sentence(
    sentence: str, bSports: bool = False, replace_symbols: tuple = (".", ",", "/", "-", "«", "»"), placeholder: str = " ", abbreviations: list = []
) -> list:
    """
        Нормализуем предложение:
        переводим символы в верхний регистр.
        удаляем служебные символы (,;!.) и пробелы по краям
        заменяем все двойные пробелы одиночными.Исключаем точки и запятые
        результат в виде списка токенов

        >normalize_sentence("", abbreviations = ["WOMEN", "(W)", "W", _
                                        "AMATEUER", "(AM)", "AM", _
                                            "RES", "(R)", "R", "B", _
                                            "VC", "BK", "BC", "SC", "SPORTING CLUB", _
                                            "FK", "FC", "FOOTBALL CLUB", "AC", "AFC", "CD", "IF", "JK", "SV", "TSV", "AL", "AL-"])
    """

    for symbol in replace_symbols:
        sentence = sentence.replace(symbol, placeholder)

    sentence = sentence.strip().upper()

    sentence = strip_doubled_characters(sentence, " ")

    if abbreviations:
        # -----------------------------------------------------------------------------------------------------------
        # Женщины,Любители+ФК и тп
        # -----------------------------------------------------------------------------------------------------------

        for term in abbreviations:
            if term in sentence:
                term_len = len(term)
                sentence = sentence.replace(" " + term + " ", " ")
                if sentence.startswith(term + " "):
                    # print(sentence ,'->', sentence[term_len + 1:])
                    sentence = sentence[term_len + 1 :]
                elif sentence.endswith(" " + term):
                    # print(sentence ,'->', sentence[:-(term_len + 1)])
                    sentence = sentence[: -(term_len + 1)]

    return list(set(sentence.split(placeholder)))


# ----------------------------------------------------------------------------------------------------------------------------
# Numba-accelerated version of sentences_similarity
# ----------------------------------------------------------------------------------------------------------------------------

try:
    import numba as nb
    import numpy as np

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
        total = N_a + N_b
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

                # Prefix match
                prefix_match = True
                for c in range(lmin_len):
                    if buf[a_start + c] != buf[b_start + c]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                if lmin_len < cMinLenTHreshold:
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
                    for k in range(l_len - s_len):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break

                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        # Greedy best-pair matching
        excluded_a = np.zeros(N_a, dtype=nb.boolean)
        excluded_b = np.zeros(N_b, dtype=nb.boolean)
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

        if w_min > 0:
            res = res / w_min * (1.0 - (w_max - w_min) / (w_max + w_min) / 5.0)

        return res

    def _pack_words(words: list) -> tuple:
        """Pack a list of strings into (buf: int32[], offsets: int32[]) for numba.

        Uses array module for fast ord() conversion instead of per-character Python loop.
        Returns (buf, offsets, n_words).
        """
        import array as _array
        n = len(words)
        parts = []
        offsets = np.empty(n + 1, dtype=np.int32)
        pos = 0
        for i, w in enumerate(words):
            offsets[i] = pos
            a = _array.array('i', w.encode('utf-32-le'))  # 4 bytes per codepoint
            parts.append(a)
            pos += len(a)
        offsets[n] = pos
        if pos > 0:
            buf = np.empty(pos, dtype=np.int32)
            p = 0
            for a in parts:
                chunk = np.frombuffer(a, dtype=np.int32)
                buf[p:p + len(chunk)] = chunk
                p += len(chunk)
        else:
            buf = np.empty(0, dtype=np.int32)
        return buf, offsets, n

    def pack_sentence(words: list) -> tuple:
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

    def sentences_similarity_numba(SentenceA: list, SentenceB: list, cMinLenTHreshold: int = 1) -> float:
        """
        Numba-accelerated version of sentences_similarity.

        Same algorithm, but Levenshtein + sliding window + greedy matching all run in compiled code.
        First call triggers JIT compilation (~1-2s), subsequent calls are 5-20x faster.

            >>> sentences_similarity_numba(["HELLO", "WORLD"], ["HELLO", "WORLD"])
            1.0

            >>> sentences_similarity_numba([], ["TEST"])

        """
        N_a = len(SentenceA)
        N_b = len(SentenceB)
        if N_a < 1 or N_b < 1:
            return None

        # Fast packing via utf-32-le bulk conversion
        all_words = SentenceA + SentenceB
        buf, offsets, _ = _pack_words(all_words)

        result = _sentences_similarity_core(buf, offsets, N_a, N_b, cMinLenTHreshold)
        if result < 0:
            return None
        return result

    def sentences_similarity_numba_packed(packed_a: tuple, packed_b: tuple, cMinLenTHreshold: int = 1) -> float:
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
        offsets = np.concatenate((off_a[:n_a], off_b[:n_b] + total_a_chars,
                                  np.array([off_b[n_b] + total_a_chars], dtype=np.int32)))

        result = _sentences_similarity_core(buf, offsets, n_a, n_b, cMinLenTHreshold)
        if result < 0:
            return None
        return result

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

                prefix_match = True
                for cc in range(lmin_len):
                    if buf[a_start + cc] != buf[b_start + cc]:
                        prefix_match = False
                        break
                if prefix_match:
                    sim_res[i, j] = 0.9 + 0.1 * lmin_len / t
                    continue

                if lmin_len < cMinLenTHreshold:
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
                    for k in range(l_len - s_len):
                        d = _lev_dist_flat(buf, s_start, s_len, l_start + k, s_len)
                        if d < best_gliding:
                            best_gliding = d
                            if d == 0:
                                break
                    sim_max = 1.0 - best_gliding / s_len
                    sim_res[i, j] = 0.5 * (sim_max + sim_min)

        excluded_a = np.zeros(N_a, dtype=nb.boolean)
        excluded_b = np.zeros(N_b, dtype=nb.boolean)
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

        if w_min > 0:
            res = res / w_min * (1.0 - (w_max - w_min) / (w_max + w_min) / 5.0)
        return res

    @nb.njit(cache=True)
    def _sentences_similarity_batch_core(buf, offsets, word_counts, n_candidates, query_n, cMinLenTHreshold):
        """Compare one query against multiple candidates sequentially."""
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
        """Compare one query against multiple candidates in parallel threads."""
        results = np.empty(n_candidates, dtype=np.float64)
        for c in nb.prange(n_candidates):
            cn = word_counts[c + 1]
            if cn < 1:
                results[c] = -1.0
            else:
                results[c] = _compare_one_candidate(buf, offsets, query_n, cand_starts[c], cn, cMinLenTHreshold)
        return results

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

    def sentences_similarity_numba_batch(query_words: list, candidates: list,
                                          cMinLenTHreshold: int = 1, parallel: bool = False) -> list:
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
            raw = _sentences_similarity_batch_parallel(
                buf, offsets, wc, cand_starts, len(candidates), n_query, cMinLenTHreshold
            )
        else:
            raw = _sentences_similarity_batch_core(
                buf, offsets, wc, len(candidates), n_query, cMinLenTHreshold
            )
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

        def __init__(self, candidates: list[list[str]], cMinLenTHreshold: int = 1, parallel: bool = False):
            self.cMinLenTHreshold = cMinLenTHreshold
            self.parallel = parallel
            self.n_candidates = len(candidates)
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

            if self.parallel:
                raw = _sentences_similarity_batch_parallel(
                    buf, offsets, wc, cand_starts, self.n_candidates, n_query, self.cMinLenTHreshold
                )
            else:
                raw = _sentences_similarity_batch_core(
                    buf, offsets, wc, self.n_candidates, n_query, self.cMinLenTHreshold
                )
            return [None if v < 0 else float(v) for v in raw]

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False

    def sentences_similarity_numba(SentenceA: list, SentenceB: list, cMinLenTHreshold: int = 1) -> float:
        """Fallback to pure-Python version when numba is not installed."""
        return sentences_similarity(SentenceA, SentenceB, cMinLenTHreshold)

    def sentences_similarity_numba_packed(packed_a, packed_b, cMinLenTHreshold: int = 1) -> float:
        """Fallback to pure-Python version when numba is not installed."""
        return sentences_similarity(packed_a, packed_b, cMinLenTHreshold) if packed_a and packed_b else None

    def sentences_similarity_numba_batch(query_words: list, candidates: list,
                                          cMinLenTHreshold: int = 1, parallel: bool = False) -> list:
        """Fallback to pure-Python version when numba is not installed."""
        return [sentences_similarity(query_words, c, cMinLenTHreshold) for c in candidates]

    def pack_sentence(words: list) -> tuple:
        """Fallback — returns words as-is."""
        return tuple(words) if words else None

    class SentenceSimilarityIndex:
        """Fallback without numba — uses pure Python."""
        def __init__(self, candidates: list, cMinLenTHreshold: int = 1, parallel: bool = False):
            self.candidates = candidates
            self.cMinLenTHreshold = cMinLenTHreshold
            self.n_candidates = len(candidates)
        def query(self, query_words: list) -> list:
            return [sentences_similarity(query_words, c, self.cMinLenTHreshold) for c in self.candidates]
