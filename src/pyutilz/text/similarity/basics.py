"""Pairwise string similarity metrics and sentence normalization.

Split out of the historical flat ``pyutilz.text.similarity`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from __future__ import annotations

from ._common import Optional, levenshtein_distance, strip_doubled_characters


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


def normalize_sentence(
    sentence: str, bSports: bool = False, replace_symbols: tuple = (".", ",", "/", "-", "«", "»"), placeholder: str = " ", abbreviations: Optional[list] = None
) -> list:
    """
    bSports is accepted and ignored: sports-specific club-suffix handling moved into the
    caller-supplied `abbreviations` list, which supersedes it. Kept in the signature so existing
    positional callers keep working.

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

    # dict.fromkeys, not set(): set() ordering varies with PYTHONHASHSEED, and the greedy
    # matcher's >= tie-break is order-sensitive, so the same pair of names could score
    # differently in two processes. Empty tokens are dropped rather than compared.
    return [token for token in dict.fromkeys(sentence.split(placeholder)) if token]
