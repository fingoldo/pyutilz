# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

ensure_installed("jellyfish numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from jellyfish import levenshtein_distance
from .strings import strip_doubled_characters
import numpy as np


def levenshtein_strings_similarity(a: str, b: str) -> float:
    """
        simple string percent similarity
        
        >>>levenshtein_strings_similarity("MeasureOIS21", "MeasureOIS18")
        0.8333333333333334
        
    """
    return 1 - levenshtein_distance(a, b) / max(len(a), len(b))


def contigous_strings_similarity(a: str, b: str) -> tuple:
    """
        similarity of 2 strings measured in contigous blocks either from the left, from the right, or both (non-symmetrically).
        it's more strong than just levenshtein_distance.
        
        >>>contigous_strings_similarity("MeosureOIS21qwe", "MeasureOIS18qwe")
        (0.3333333333333333, 'Meqwe')
        
        >>>contigous_strings_similarity("MeosureOIS21qwe", "MeosureOIS21qwe")
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
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Улучшенный алгоритм распознавания одинаковых фраз, в т.ч. спортивных команд/игроков, адресов
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    # Dim MaxFullSim!
    # Dim Pairs_left&                                                     'Нестыкованных пар слов
    # Dim last_b_ind&                                                     'Для названия команд из одного слова
    # Dim N_a&, N_b&, sz&                                                 'Слов в имени команды/игрока
    # Dim best_gliding_perf&                                              'Наименьшая скользящая непохожесть ;-)
    # Dim best_perf!, best_i&, best_j&                                    'Лучшая параDim
    # Dim i&, j&, k&, t&, w_min&, w_max&, cur_a_len&, cur_b_len&, t_sim&

    # Для каждой пары слов из А,В рассчитывам максимальную и минимальную похожести.
    # Минимальная - это просто метрическое расстояние.
    # При неодинаковых длинах слов максимальная похожесть отличается от минимальной, и представляет собой лучший результат сравнения при скольжении меньшего слова вдоль большего

    simN_a = 0
    simN_b = 0
    last_b_ind = -1
    res = 0.0

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

    bSim_A_Excluded = np.zeros(N_a)
    bSim_B_Excluded = np.zeros(N_b)
    Sim_max_l = np.zeros((N_a, N_b))
    Sim_min_l = np.zeros((N_a, N_b))
    Sim_res_l = np.zeros((N_a, N_b))

    if simN_b != N_b:
        simN_b = N_b

    if simN_a != N_a:
        simN_a = N_a

    for i in range(N_a):
        cur_a_len = len(SentenceA[i])
        if cur_a_len > 0:
            a = SentenceA[i]
            for j in range(N_b):

                if j != last_b_ind:
                    b = SentenceB[j]
                    cur_b_len = len(b)

                    last_b_ind = j
                # ----------------------------------------------------------------------------------------------------------------------------------------------------
                if cur_b_len > 0:
                    t = max(cur_a_len, cur_b_len)
                    # ----------------------------------------------------------------------------------------------------------------------------------------------------
                    # if one of words is exact substing of another word
                    # ----------------------------------------------------------------------------------------------------------------------------------------------------

                    bStartsMatched = False
                    lminLen = min(cur_a_len, cur_b_len)
                    if lminLen >= 1:
                        if a[:lminLen] == b[:lminLen]:
                            bStartsMatched = True
                            if cur_a_len == cur_b_len:
                                Sim_min_l[i, j] = 1
                                Sim_max_l[i, j] = 1
                            else:
                                # Чтобы "Almeria B"<>"Al-Budaiya"
                                Sim_min_l[i, j] = 0.9 + 0.1 * lminLen / t
                                Sim_max_l[i, j] = Sim_min_l[i, j]

                    if not bStartsMatched:
                        if lminLen < cMinLenTHreshold:
                            Sim_min_l[i, j] = 0
                            Sim_max_l[i, j] = 0
                        else:
                            Sim_min_l[i, j] = 1 - levenshtein_distance(a, b) / max(cur_a_len, cur_b_len)
                            if cur_a_len == cur_b_len:
                                Sim_max_l[i, j] = Sim_min_l[i, j]
                            else:
                                # Скольжение
                                best_gliding_perf = 1000
                                if cur_a_len < cur_b_len:

                                    for k in range(cur_b_len - cur_a_len):
                                        t_sim = levenshtein_distance(a, b[k : k + cur_a_len])
                                        if t_sim < best_gliding_perf:
                                            best_gliding_perf = t_sim

                                else:
                                    c = a[:cur_b_len]

                                    for k in range(cur_a_len - cur_b_len):
                                        t_sim = levenshtein_distance(b, a[k : k + cur_b_len])
                                        if t_sim < best_gliding_perf:
                                            best_gliding_perf = t_sim

                                # Нашли самый лучший вариант
                                Sim_max_l[i, j] = 1 - best_gliding_perf / min(cur_a_len, cur_b_len)

                    #
                    # рассчитываем для каждой пары коэффициент l=Sim_max_l+Sim_min_l/100
                    Sim_res_l[i, j] = 0.5 * (Sim_max_l[i, j] + Sim_min_l[i, j])

    # начиная с самых больших значений l, вычёркиваем пары слов из дальнейшего рассмотрения. при этом сохраняем в списке их Sim_max_l
    Pairs_left = w_min
    MaxFullSim = -1

    # print(Sim_res_l)

    while Pairs_left != 0:
        # Находим самую похожую пару слов из неисключённых
        best_perf = 0
        best_i = 0
        best_j = 0
        for i in range(N_a):
            if not bSim_A_Excluded[i]:
                for j in range(N_b):
                    if not bSim_B_Excluded[j]:
                        if Sim_res_l[i, j] >= best_perf:
                            best_perf = Sim_res_l[i, j]
                            best_i = i
                            best_j = j
                            if Sim_min_l[i, j] > MaxFullSim:
                                MaxFullSim = Sim_res_l[i, j]

        # Теперь учитываем эту пару в результате
        # средневзвешенная схожесть имён (а точнее, слов в фразе) рассчитывается взвешиванием 1/Huracan (Arg) по Sim_max_l, где w_min - минимальное из количества слов.
        res = res + Sim_res_l[best_i, best_j]
        # И исключаем слова из дальнейшей работы
        bSim_A_Excluded[best_i] = True
        bSim_B_Excluded[best_j] = True
        Pairs_left = Pairs_left - 1

    if w_min > 0:
        res = res / w_min * (1 - (w_max - w_min) / (w_max + w_min) / 5)
    else:
        res = 0

    #    if MaxFullSim < cMinSimilarityLevel:
    #        res = 0
    #

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
                l = len(term)
                sentence = sentence.replace(" " + term + " ", " ")
                if sentence.startswith(term + " "):
                    # print(sentence ,'->', sentence[l + 1:])
                    sentence = sentence[l + 1 :]
                elif sentence.endswith(" " + term):
                    # print(sentence ,'->', sentence[:-(l + 1)])
                    sentence = sentence[: -(l + 1)]

    return list(set(sentence.split(placeholder)))
