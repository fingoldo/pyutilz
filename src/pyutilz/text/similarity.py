"""String and sentence similarity metrics: Levenshtein-based string similarity, word-overlap sentence similarity (with numba-accelerated variants), and a batched candidate-matching index."""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


from jellyfish import levenshtein_distance
from pyutilz.text.strings import strip_doubled_characters
from typing import Optional

# Above this many tokens per sentence, the O(w_min * N_a * N_b) greedy matching pass
# (a full rescan of the similarity matrix for each of the w_min greedy picks, on top of the
# O(N_a * N_b) matrix fill) starts to dominate. Benchmarked: N=10 -> ~0.3ms, N=80 -> ~25-30ms
# per call (pure-Python path). Not severe enough to justify a heap-based rewrite (which would
# need a differential-correctness test against the current tie-breaking to avoid behavior
# drift), but callers feeding longer inputs should be aware of the quadratic-ish cost.
SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD = 50

_VALID_COVERAGE_SIDES = ("min", "max", "both")


def _validate_coverage_side(coverage_side: str) -> None:
    """Raise ValueError unless `coverage_side` is one of `_VALID_COVERAGE_SIDES` ("min"/"max"/"both")."""
    if coverage_side not in _VALID_COVERAGE_SIDES:
        raise ValueError(f"coverage_side must be one of {_VALID_COVERAGE_SIDES}, got {coverage_side!r}")


def _check_word_coverage(
    words: list,
    matched_sims,
    min_word_similarity: float,
    required_coverage: Optional[float],
    required_matched_words: Optional[int],
    stop_set: set,
) -> bool:
    """True iff enough of `words`' non-stopword entries have an achieved match similarity
    (`matched_sims`, same length/order as `words`, 0.0 where a word went unmatched - always true for
    every word on the w_max side of a greedy w_min-pair match) at or above `min_word_similarity`.

    A word not in `stop_set` is "content"; coverage is judged only over content words - a sentence
    made entirely of stopwords (or empty) trivially passes, since there is nothing to require
    coverage of. Both `required_matched_words` (absolute count) and `required_coverage` (fraction of
    content words) may be supplied together; both must pass.
    """
    content_indices = [i for i, w in enumerate(words) if w not in stop_set]
    if not content_indices:
        return True
    covered = sum(1 for i in content_indices if matched_sims[i] >= min_word_similarity)
    if required_matched_words is not None and covered < required_matched_words:
        return False
    if required_coverage is not None and covered / len(content_indices) < required_coverage:
        return False
    return True


def _strip_stop_words(words: list, stop_words: Optional[list]) -> list:
    """`words` with every entry in `stop_words` removed, or `words` unchanged when that would empty the
    list (a sentence made entirely of stop words has nothing else to score against) or when `stop_words`
    is falsy.

    Applied BEFORE the w_max length-normalization (2026-08-12's own fix) sees either sentence, not only
    inside the opt-in coverage gate `_check_word_coverage` already excludes stop words from: a connective
    word ("some", "I", "have") still counted as query LENGTH `w_max` couldn't explain, silently dragging a
    correct single-word match below any reasonable score threshold - measured live in autopsia's complaint
    parser, 2026-08-16 ("some nausea" scored 0.0 against "Nausea", "I have nausea" scored 0.56, both used
    to score 1.0 before the w_max fix). `stop_words` was already a real, threaded parameter every call site
    below accepts - this widens what it does (base score, not just the coverage gate), it does not add new
    API surface."""
    if not stop_words:
        return words
    stop_set = set(stop_words)
    filtered = [w for w in words if w not in stop_set]
    if not filtered:
        return words
    return filtered


def levenshtein_strings_similarity(a: str, b: str) -> float:
    """
    simple string percent similarity

    >>> levenshtein_strings_similarity("MeasureOIS21", "MeasureOIS18")
    0.8333333333333334

    >>> levenshtein_strings_similarity("", "")
    1.0

    """
    if not a and not b:
        return 1.0
    return 1 - levenshtein_distance(a, b) / max(len(a), len(b))


def contigous_strings_similarity(a: str, b: str) -> tuple:
    """
    similarity of 2 strings measured in contiguous blocks either from the left, from the right, or both (non-symmetrically).
    it's more strong than just levenshtein_distance.

    >>> contigous_strings_similarity("MeosureOIS21qwe", "MeasureOIS18qwe")
    (0.3333333333333333, 'Meqwe')

    >>> contigous_strings_similarity("MeosureOIS21qwe", "MeosureOIS21qwe")
    (1.0, 'MeosureOIS21qwe')

    >>> contigous_strings_similarity("", "")
    (1.0, '')
    """
    if not a and not b:
        return 1.0, ""

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


def sentences_similarity(
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

    Complexity: O(N_a * N_b) to fill the similarity matrix, plus O(w_min * N_a * N_b) for the
    greedy best-pair matching pass (a full matrix rescan per pick) — effectively O(w * N^2)
    for near-square inputs. This is fine for the documented use case (short phrases, team/player
    names, addresses — typically 3-10 tokens); on longer inputs it degrades quadratically-ish
    (benchmarked ~25-30ms at N=80 per call on the pure-Python path). Avoid using this function
    on sentences longer than ~20-30 tokens; a warning is logged past
    SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD (50) tokens.

    Optional coverage gate (opt-in, off by default — zero overhead unless `required_coverage` or
    `required_matched_words` is set): requires enough non-stopword words on the specified
    `coverage_side` ("min"/"max"/"both") to have an achieved greedy-match similarity of at least
    `min_word_similarity`, else returns None instead of a score. Guards against a candidate that
    "wins" purely on one strong word while leaving the rest of the sentence unaccounted for — e.g.
    a 1-word candidate ("опущение") matching one word of a 2-word query ("опущение века") and
    silently discarding the other, real, meaning-changing word ("века"/eyelid) - measured live in
    autopsia's RU symptom resolver, 2026-08-12 ("опущение века" resolving to the wrong organ system
    entirely via that dropped word). `stop_words`, if given, excludes connective words (prepositions,
    conjunctions) from the coverage requirement entirely - they never had a chance to score a
    meaningful match and forcing coverage of them would make the gate too strict for ordinary phrasing.

        >>> sentences_similarity(["A"], ["A", "B"], required_coverage=1.0)

        >>> sentences_similarity(["A", "B"], ["A", "B"], required_coverage=1.0)
        1.0
    """
    SentenceA = _strip_stop_words(SentenceA, stop_words)
    SentenceB = _strip_stop_words(SentenceB, stop_words)
    N_a = len(SentenceA)
    N_b = len(SentenceB)
    if N_a < 1 or N_b < 1:
        return None
    coverage_active = required_coverage is not None or required_matched_words is not None
    if coverage_active:
        _validate_coverage_side(coverage_side)
    if N_a > SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD or N_b > SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD:
        logger.warning(
            "sentences_similarity called with N_a=%d, N_b=%d tokens, exceeding the safe threshold of %d. "
            "The greedy matching pass is O(w_min * N_a * N_b); expect quadratic-ish slowdown on long inputs.",
            N_a,
            N_b,
            SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD,
        )
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
    # Per-word achieved match similarity (0.0 for any word left unmatched after w_min picks - true
    # for every word on the w_max side by construction, since the greedy pass never picks more than
    # w_min pairs). Only READ when the coverage gate is active; the O(N) allocation/writes cost
    # nothing next to the O(w*N^2) matching pass either way, so no need to special-case them off.
    matched_sim_a: list = [0.0] * N_a
    matched_sim_b: list = [0.0] * N_b

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
        matched_sim_a[best_i] = Sim_res[best_i][best_j]
        matched_sim_b[best_j] = Sim_res[best_i][best_j]
        excluded_a[best_i] = True
        excluded_b[best_j] = True

    # Normalized by w_max (the LONGER sentence), not w_min: the greedy pass above only ever picks
    # w_min pairs, so a short candidate needs to explain few words to average high under a w_min
    # denominator - one lucky exact match can dominate the whole score, even when the candidate
    # leaves most of a longer query unexplained. Measured live (autopsia's RU symptom-resolution
    # fuzzy matcher, 2026-08-12): "слабость века" (2 words, one exact hit on "слабость") scored
    # HIGHER than "слабость мышц ног" (3 words, the actually-correct, more specific match) against
    # the query "мышечная слабость в ногах" - confirmed structural, not a one-off, by a
    # leave-one-out calibration over 700+ dictionary entries (false-top-match mean 0.847 vs
    # true-pair mean 0.593, no separating threshold existed). Dividing by w_max instead makes each
    # unmatched word on the longer side count as an implicit zero contribution to the average, so a
    # short candidate can never out-score a longer, more specific one just by leaving more of the
    # query unaccounted for - the previous multiplicative length-penalty term is now redundant
    # (this normalization already encodes it, exactly rather than via a bounded ad-hoc fudge) and
    # is removed. For EQUAL-length inputs (w_min == w_max, the documented primary use case - team
    # names, addresses) this is a no-op: identical to the old formula.
    if w_max > 0:
        res = res / w_max
    else:
        res = 0

    if coverage_active:
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

    return res


def normalize_sentence(
    sentence: str, bSports: bool = False, replace_symbols: tuple = (".", ",", "/", "-", "«", "»"), placeholder: str = " ", abbreviations: Optional[list] = None
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
    if abbreviations is None:
        abbreviations = []

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

    def sentences_similarity_numba_packed(packed_a: tuple, packed_b: tuple, cMinLenTHreshold: int = 1) -> Optional[float]:
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
        for c in nb.prange(n_candidates):  # type: ignore[attr-defined]
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

            # Coverage-active path: needs per-candidate word-level match tracking, which the batch
            # kernels above don't expose (scalar-only) - loop per candidate instead. Not parallelized
            # (correctness-first for an opt-in feature); the default path above is unaffected.
            stop_set = set(self.stop_words) if self.stop_words else set()
            results: list = []
            for idx in range(self.n_candidates):
                cn = int(self._wc_arr[idx])
                if cn < 1:
                    results.append(None)
                    continue
                res, matched_sim_a, matched_sim_b = _compare_one_candidate_with_matches(buf, offsets, n_query, int(cand_starts[idx]), cn, self.cMinLenTHreshold)
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

    HAS_NUMBA = True

except ImportError:
    HAS_NUMBA = False

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

    def sentences_similarity_numba_packed(packed_a: tuple, packed_b: tuple, cMinLenTHreshold: int = 1) -> Optional[float]:
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
