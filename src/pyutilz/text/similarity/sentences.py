"""Word-overlap sentence similarity (pure-Python reference implementation).

Split out of the historical flat ``pyutilz.text.similarity`` module; re-exported
from the package ``__init__`` to preserve the public import surface. The numba
variants of this function live in ``_numba_api``/``_numba_kernels``.
"""

from __future__ import annotations

from ._common import (
    Optional,
    SENTENCES_SIMILARITY_SAFE_TOKEN_THRESHOLD,
    _check_word_coverage,
    _strip_stop_words,
    _validate_coverage_side,
    levenshtein_distance,
    logger,
)


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

            if lminLen < cMinLenTHreshold:
                continue

            # Check if starts match (one is prefix of another)
            # Чтобы "Almeria B"<>"Al-Budaiya": a single shared character must not earn the 0.9
            # floor -- "A" vs "ANDERSON" scored 0.9125, above a correct suffix match.
            if lminLen >= 2 and a[:lminLen] == b[:lminLen]:
                sim = 0.9 + 0.1 * lminLen / t
                Sim_res[i][j] = sim
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
                # L-S+1 windows of length S exist; the old `range(L-S)` never tested the SUFFIX window,
                # so an exact suffix match (MADRID vs REALMADRID) scored as unrelated.
                for k in range(long_len - short_len + 1):
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
