"""Mutation-survivor pins for humanizer: the exact replacement tables, typo branches, span remapping and recheck survivors."""

import random
import pytest
_AI_PATTERN_REPLACEMENTS = [
    ("We think it's worth noting that tests pass.", "We think tests pass."),
    ("We think it is important to note that tests pass.", "We think tests pass."),
    ("We have, in conclusion, tests.", "We have, tests."),
    ("We did, to summarize, the tests.", "We did, the tests."),
    ("This stands as a testament to care.", "This shows to care."),
    ("This underscores its importance here.", "This matters here."),
    ("We, as an AI, ship it.", "We, ship it."),
    ("We delve into data.", "We look into data."),
    ("We delve deeper.", "We dig deeper."),
    ("We leverage tools.", "We use tools."),
    ("We utilize tools.", "We use tools."),
    ("We facilitate work.", "We help with work."),
    ("We streamline work.", "We simplify work."),
    ("We need comprehensive tests.", "We need thorough tests."),
    ("We ensure that tests pass.", "We make sure tests pass."),
    ("We ensure tests pass.", "We make sure tests pass."),
    ("A pivotal change.", "A key change."),
    ("A seamless flow.", "A smooth flow."),
    ("A robust design.", "A solid design."),
    ("A meticulous review.", "A careful review."),
    ("This innovative idea.", "This new idea."),
    ("A cutting-edge tool.", "A modern tool."),
    ("A cutting edge tool.", "A modern tool."),
    ("Tests pass. Furthermore, we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Moreover, we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Additionally we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Consequently we ship.", "Tests pass. So, we ship."),
    ("Tests pass. Nevertheless we ship.", "Tests pass. Still, we ship."),
    ("I'm thrilled to help.", "I'm glad to help."),
    ("I'd love to help.", "I'd like to help."),
    ("I bring 10 years of experience.", "I have 10 years of experience."),
    ("This plays a vital role here.", "This is important here."),
]
_GLUED_NEGATIVES = [
    "xIt is important to note that tests stay.",
    "xIn conclusion, tests stay.",
    "xTo summarize, tests stay.",
    "xstands as a testament to care.",
    "This underscores itsimportance here.",
    "xdelve into data stay.",
    "xdelve deeper and stay.",
    "xleverage tools and stay.",
    "xutilize tools and stay.",
    "xfacilitate work and stay.",
    "xstreamline work and stay.",
    "xcomprehensive tests stay.",
    "xensure that tests stay.",
    "xensure tests stay.",
    "xpivotal change stays.",
    "xseamless flow stays.",
    "xmeticulous review stays.",
    "xcutting edge tool stays.",
    "We xFurthermore, we stay.",
    "xI'm thrilled to help and stay.",
    "xI'd love to help and stay.",
    "xI bring 10 years of work and stay.",
    "This plays avital role here.",
    "xplays a vital role here and stay.",
]


class _Script:
    """A stand-in for random.Random that returns the values a test queues, in order, and checks each choice is legal."""

    def __init__(self, *values):
        self._values = list(values)

    def _next(self):
        return self._values.pop(0)

    def choice(self, seq):
        value = self._next()
        assert value in seq, (value, seq)
        return value

    def randint(self, lo, hi):
        value = self._next()
        assert lo <= value <= hi, (value, lo, hi)
        return value

    def choices(self, population, weights=None, k=1):
        return [self._next()]

_AI_PATTERN_REPLACEMENTS = [
    ("We think it's worth noting that tests pass.", "We think tests pass."),
    ("We think it is important to note that tests pass.", "We think tests pass."),
    ("We have, in conclusion, tests.", "We have, tests."),
    ("We did, to summarize, the tests.", "We did, the tests."),
    ("This stands as a testament to care.", "This shows to care."),
    ("This underscores its importance here.", "This matters here."),
    ("We, as an AI, ship it.", "We, ship it."),
    ("We delve into data.", "We look into data."),
    ("We delve deeper.", "We dig deeper."),
    ("We leverage tools.", "We use tools."),
    ("We utilize tools.", "We use tools."),
    ("We facilitate work.", "We help with work."),
    ("We streamline work.", "We simplify work."),
    ("We need comprehensive tests.", "We need thorough tests."),
    ("We ensure that tests pass.", "We make sure tests pass."),
    ("We ensure tests pass.", "We make sure tests pass."),
    ("A pivotal change.", "A key change."),
    ("A seamless flow.", "A smooth flow."),
    ("A robust design.", "A solid design."),
    ("A meticulous review.", "A careful review."),
    ("This innovative idea.", "This new idea."),
    ("A cutting-edge tool.", "A modern tool."),
    ("A cutting edge tool.", "A modern tool."),
    ("Tests pass. Furthermore, we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Moreover, we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Additionally we ship.", "Tests pass. Also, we ship."),
    ("Tests pass. Consequently we ship.", "Tests pass. So, we ship."),
    ("Tests pass. Nevertheless we ship.", "Tests pass. Still, we ship."),
    ("I'm thrilled to help.", "I'm glad to help."),
    ("I'd love to help.", "I'd like to help."),
    ("I bring 10 years of experience.", "I have 10 years of experience."),
    ("This plays a vital role here.", "This is important here."),
]

_GLUED_NEGATIVES = [
    "xIt is important to note that tests stay.",
    "xIn conclusion, tests stay.",
    "xTo summarize, tests stay.",
    "xstands as a testament to care.",
    "This underscores itsimportance here.",
    "xdelve into data stay.",
    "xdelve deeper and stay.",
    "xleverage tools and stay.",
    "xutilize tools and stay.",
    "xfacilitate work and stay.",
    "xstreamline work and stay.",
    "xcomprehensive tests stay.",
    "xensure that tests stay.",
    "xensure tests stay.",
    "xpivotal change stays.",
    "xseamless flow stays.",
    "xmeticulous review stays.",
    "xcutting edge tool stays.",
    "We xFurthermore, we stay.",
    "xI'm thrilled to help and stay.",
    "xI'd love to help and stay.",
    "xI bring 10 years of work and stay.",
    "This plays avital role here.",
    "xplays a vital role here and stay.",
]

class _Recorder(_Script):
    """_Script that also records the bounds of every randint call and the candidate list of every choice call."""

    def __init__(self, *values):
        super().__init__(*values)
        self.ranges: list = []
        self.seqs: list = []

    def randint(self, lo, hi):
        self.ranges.append((lo, hi))
        return super().randint(lo, hi)

    def choice(self, seq):
        self.seqs.append(list(seq))
        return super().choice(seq)


class TestTablesAndAdjacencyPinned:
    """The keyboard neighbour tables are pinned whole, the typo weights sum to one, and the case of an adjacent key follows
    the typed character (humanizer.py 22-52, 53-65, 300-317)."""

    def test_the_qwerty_neighbour_table_is_pinned_whole(self):
        from pyutilz.text import humanizer as h

        assert h._QWERTY_ADJACENT == {
            "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
            "y": "tugh", "u": "yijh", "i": "uojk", "o": "iplk", "p": "ol",
            "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
            "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "iolmj",
            "l": "opk",
            "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
            "b": "vghn", "n": "bhjm", "m": "njk",
        }  # fmt: skip

    def test_the_yцукен_neighbour_table_is_pinned_whole(self):
        from pyutilz.text import humanizer as h

        assert h._YCUKEN_ADJACENT == {
            "й": "цф", "ц": "йуфыв", "у": "цквыа", "к": "уеапр",
            "е": "кнпро", "н": "егрол", "г": "нщолд", "ш": "щзлдж",
            "щ": "гшздж", "з": "шщжхэ", "х": "зэж", "ъ": "хэ",
            "ф": "йцыя", "ы": "цуфвач", "в": "укыапс", "а": "кевпрм",
            "п": "еналри", "р": "нгпоит", "о": "гшрлдь", "л": "щгодж",
            "д": "шщлжэб", "ж": "зхдэю", "э": "хъжю",
            "я": "фыч", "ч": "ыясв", "с": "вчмап", "м": "аспи",
            "и": "прмт", "т": "ориьб", "ь": "олтбю", "б": "лдьюж",
            "ю": "джэб",
        }  # fmt: skip

    def test_the_typo_weights_are_pinned_and_sum_to_one(self):
        from pyutilz.text import humanizer as h

        assert h._TYPO_TYPES == [
            ("adjacent_key", 0.30),
            ("transpose", 0.20),
            ("double_letter", 0.15),
            ("skip_letter", 0.10),
            ("extra_space", 0.10),
            ("missing_space_comma", 0.10),
            ("missing_space_period", 0.05),
        ]
        assert abs(sum(w for _, w in h._TYPO_TYPES) - 1.0) < 1e-12

    def test_an_upper_case_letter_gets_upper_case_neighbours(self):
        from pyutilz.text.humanizer import _get_adjacent

        assert _get_adjacent("Q") == "WA"
        assert _get_adjacent("й") == "цф"
        assert _get_adjacent("Й") == "ЦФ"
        assert _get_adjacent("1") is None

class TestTypoBranchesWithScriptedChoices:
    """Each character and spacing typo, driven by queued choices so the exact edit is known (humanizer.py 320-447)."""

    def test_a_long_word_is_the_only_target_when_the_first_and_last_words_are_skipped(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        words = [(0, "start"), (6, "there"), (12, "end")]
        assert _pick_word_for_typo(words, (), _Script((6, "there"))) == (6, "there")

    def test_a_word_shorter_than_the_minimum_is_never_a_target(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        words = [(0, "aa"), (3, "bbbb"), (8, "ccccc"), (14, "dd")]
        assert _pick_word_for_typo(words, (), _Script((8, "ccccc"))) == (8, "ccccc")

    def test_a_capitalised_word_is_skipped_but_an_all_capitals_word_is_not(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        words = [(0, "start"), (6, "Hello"), (12, "WORLD"), (18, "end")]
        assert _pick_word_for_typo(words, (), _Script((12, "WORLD"))) == (12, "WORLD")

    def test_a_protected_word_is_skipped_and_an_unprotected_neighbour_is_used(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        words = [(0, "start"), (6, "alphabet"), (15, "middle"), (22, "end")]
        assert _pick_word_for_typo(words, [(6, 14)], _Script((15, "middle"))) == (15, "middle")

    def test_no_eligible_word_means_no_target(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        assert _pick_word_for_typo([(0, "a"), (2, "b"), (4, "c")], (), random.Random(0)) is None

    def test_an_adjacent_key_typo_replaces_one_letter_with_a_neighbour(self):
        from pyutilz.text.humanizer import _apply_char_typo

        out, pos, delta = _apply_char_typo("hello there world", "adjacent_key", (), _Script((6, "there"), 0, "y"))
        assert out == "hello yhere world"
        assert (pos, delta) == (6, 0)

    def test_a_transposition_swaps_two_inner_letters(self):
        from pyutilz.text.humanizer import _apply_char_typo

        out, pos, delta = _apply_char_typo("hello there world", "transpose", (), _Script((6, "there"), 2))
        assert out == "hello three world"
        assert (pos, delta) == (6, 0)

    def test_a_double_letter_repeats_an_inner_letter(self):
        from pyutilz.text.humanizer import _apply_char_typo

        out, pos, delta = _apply_char_typo("hello there world", "double_letter", (), _Script((6, "there"), 1))
        assert out == "hello thhere world"  # codespell:ignore thhere
        assert (pos, delta) == (6, 1)

    def test_a_skipped_letter_drops_an_inner_letter(self):
        from pyutilz.text.humanizer import _apply_char_typo

        out, pos, delta = _apply_char_typo("hello there world", "skip_letter", (), _Script((6, "there"), 2))
        assert out == "hello thre world"  # codespell:ignore thre
        assert (pos, delta) == (6, -1)

    def test_an_unknown_typo_kind_leaves_the_text_alone(self):
        from pyutilz.text.humanizer import _apply_char_typo

        assert _apply_char_typo("hello there world", "mystery", (), _Script((6, "there"))) == ("hello there world", 0, 0)

    def test_an_extra_space_doubles_one_inter_word_gap(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("a b c", "extra_space", (), _Script(3)) == ("a b  c", 3, 1)

    def test_a_missing_space_after_a_comma_is_removed(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("one, two", "missing_space_comma", (), _Script(3)) == ("one,two", 4, -1)

    def test_a_missing_space_after_a_sentence_end_is_removed_before_a_capital(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("Done. Next", "missing_space_period", (), _Script(4)) == ("Done.Next", 5, -1)

    def test_a_space_typo_with_nothing_to_hit_leaves_the_text_alone(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("oneword", "extra_space", (), random.Random(0)) == ("oneword", 0, 0)
        assert _apply_space_typo("no comma here", "missing_space_comma", (), random.Random(0)) == ("no comma here", 0, 0)

class TestDashesQuotesAndEmojiExactly:
    """fix_dashes and strip_emojis, pinned on exact outputs (humanizer.py 235-315)."""

    def test_an_em_dash_becomes_a_spaced_hyphen(self):
        from pyutilz.text.humanizer import fix_dashes

        assert fix_dashes("a—b") == "a - b"

    def test_an_en_dash_becomes_a_spaced_hyphen(self):
        from pyutilz.text.humanizer import fix_dashes

        assert fix_dashes("a–b") == "a - b"

    def test_a_dash_padded_by_spaces_is_collapsed_to_one_spaced_hyphen(self):
        from pyutilz.text.humanizer import fix_dashes

        assert fix_dashes("a — b") == "a - b"
        assert fix_dashes("a -- b") == "a - b"

    def test_each_smart_quote_becomes_its_ascii_form(self):
        from pyutilz.text.humanizer import fix_dashes

        assert fix_dashes("“q”") == '"q"'
        assert fix_dashes("it’s ‘fine’") == "it's 'fine'"

    def test_layout_spaces_with_no_dash_are_left_alone(self):
        from pyutilz.text.humanizer import fix_dashes

        assert fix_dashes("col1    col2") == "col1    col2"

    def test_an_emoji_is_removed_and_the_ends_are_trimmed(self):
        from pyutilz.text.humanizer import strip_emojis

        assert strip_emojis("Hi \U0001f600 there") == "Hi  there"
        assert strip_emojis("done \U0001f680") == "done"

class TestCapitalAndPipelineBehaviour:
    """strip_ai_patterns restores a capital only when the text started with one, and humanize keeps the typo count, the
    spans and the warning exactly (humanizer.py 182-233, 449-563)."""

    def test_a_removed_opener_recapitalises_the_text_when_it_started_capitalised(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("It is worth noting that we shipped it.") == "We shipped it."

    def test_a_text_that_started_lower_case_is_not_capitalised(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("it is worth noting that we shipped it.") == "we shipped it."

    def test_a_vocabulary_downgrade_at_the_start_is_recapitalised(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("Robust pipelines matter.") == "Solid pipelines matter."

    def test_zero_typos_returns_the_cleaned_text_without_touching_words(self):
        from pyutilz.text.humanizer import humanize

        assert humanize("We leverage tools — safely.", typo_count=0) == "We use tools - safely."

    def test_a_span_that_cleaning_removed_is_reported_once_and_the_rest_still_applies(self, caplog):
        import logging

        from pyutilz.text.humanizer import humanize

        text = "It is worth noting that alpha."
        with caplog.at_level(logging.WARNING, logger="pyutilz.text.humanizer"):
            out = humanize(text, typo_count=0, protected_spans=[(6, 23)])
        assert out == "Alpha."
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == ["humanize(): 1 protected span(s) not found after cleaning -- NOT protected from typo injection: ['worth noting that']"]

    def test_an_empty_span_is_ignored(self):
        from pyutilz.text.humanizer import humanize

        assert humanize("Alpha beta", typo_count=0, protected_spans=[(3, 3)]) == "Alpha beta"

@pytest.mark.parametrize(("text", "expected"), _AI_PATTERN_REPLACEMENTS)
def test_each_filler_vocabulary_and_transition_rule_gives_its_exact_replacement(text, expected):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == expected

@pytest.mark.parametrize(
    "text",
    [
        "The nonrobust path stays.",
        "The undelve plan stays.",
        "The reensure plan stays.",
        "The noncomprehensive plan stays.",
        "The noninnovative plan stays.",
        "The unpivotal plan stays.",
        "They delveinto data and stay.",
        "We ensurethat tests stay.",
        "I bring years of work and stay.",
        "WIt's worth noting that tests stay.",
    ],
)
def test_a_word_inside_a_longer_word_or_a_glued_phrase_is_never_rewritten(text):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == text

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Use the loop (more idiomatic) here.", "Use the loop here."),
        ("Use it (which is cleaner).", "Use it."),
        ("Use it (cleaner way).", "Use it."),
        ("Use it (more idiomatic approach) now.", "Use it now."),
        ("Use it (the robust variant ships in 3.12).", "Use it (the solid variant ships in 3.12)."),
    ],
)
def test_a_parenthetical_self_justification_is_removed_only_when_it_is_nothing_else(text, expected):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == expected

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sure thing! We ship.", "We ship."),
        ("Happy to help. We ship.", "We ship."),
        ("Basically, we ship.", "We ship."),
        ("Fundamentally, we ship.", "We ship."),
    ],
)
def test_each_hedging_or_filler_opener_is_removed_and_the_capital_restored(text, expected):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == expected

@pytest.mark.parametrize(
    "char",
    [
        "\U0001f300", "\U0001f680", "\U0001f900", "\U0001fa00", "\U0001fa70",
        "\u2715", "\U0001f1e6", "\u20e3", "\ufe0f", "\u200d",
        "\u231a", "\u23e9", "\u23f8", "\u25aa", "\u25b6", "\u25c0", "\u25fb",
        "\u2934", "\u2b05", "\u2b1b", "\u2b50", "\u2b55", "\u3030", "\u303d",
        "\u3297", "\u3299",
    ],
)
def test_each_emoji_range_is_stripped_by_its_own_representative(char):
    from pyutilz.text.humanizer import strip_emojis

    assert strip_emojis(f"a{char}b") == "ab"

@pytest.mark.parametrize("mark", ["\u2713", "\u2714"])
def test_the_checkmarks_that_a_checklist_uses_are_kept(mark):
    from pyutilz.text.humanizer import strip_emojis

    assert strip_emojis(f"{mark} done") == f"{mark} done"

@pytest.mark.parametrize("text", _GLUED_NEGATIVES)
def test_a_rule_does_not_fire_inside_a_glued_word_or_a_broken_phrase(text):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == text

def test_a_double_space_before_the_number_is_not_a_bring_years_phrase():
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns("I bring  years of work.") == "I bring years of work."
    assert strip_ai_patterns("I bring years of work.") == "I bring years of work."

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Great question. We ship.", "We ship."),
        ("Excellent question. We ship.", "We ship."),
        ("Good question. We ship.", "We ship."),
    ],
)
def test_each_question_praise_opener_is_removed(text, expected):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == expected

def test_a_filler_opener_needs_a_space_after_the_sentence_end():
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns("Done.Basically, we ship.") == "Done.Basically, we ship."
    assert strip_ai_patterns("Done. Basically, we ship.") == "Done. We ship."

@pytest.mark.parametrize(
    "text",
    [
        "xMoreover, we stay.",
        "xAdditionally we stay.",
        "xConsequently we stay.",
        "xNevertheless we stay.",
        "This underscores itsimportance here.",
        "xAs an AI, ship it.",
    ],
)
def test_a_transition_or_hedge_rule_does_not_fire_glued_or_without_its_spacing(text):
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns(text) == text

def test_the_as_an_ai_phrase_keeps_its_language_model_words_when_they_are_not_separated():
    from pyutilz.text.humanizer import strip_ai_patterns

    assert strip_ai_patterns("We, as an AIlanguage model, ship it.") == "We, language model, ship it."

class TestTypoEngineBoundariesAndArguments:
    """Protected-range overlap at its edges, the word picker's first, last and capital rules, each character typo's draw
    range and resulting word, the spacing typos' protected checks, and the defaults of introduce_typos and humanize
    (humanizer.py 320-561)."""

    def test_protected_overlap_is_strict_at_both_edges(self):
        from pyutilz.text.humanizer import _is_protected

        assert _is_protected(5, 5, [(0, 5)]) is False
        assert _is_protected(0, 5, [(5, 9)]) is False
        assert _is_protected(4, 2, [(0, 5)]) is True

    def test_the_first_and_the_last_word_are_both_skipped_by_the_picker(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        assert _pick_word_for_typo([(0, "alpha"), (6, "bravo")], (), random.Random(0)) is None
        assert _pick_word_for_typo([(0, "start"), (6, "abc1def"), (14, "end")], (), random.Random(0)) is None

    def test_the_last_word_is_skipped_even_when_it_is_long_enough(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        words = [(0, "start"), (6, "there"), (12, "bravo")]
        for seed in range(20):
            assert _pick_word_for_typo(words, (), random.Random(seed)) == (6, "there")

    def test_a_capital_inside_a_word_does_not_make_it_a_proper_noun(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        assert _pick_word_for_typo([(0, "start"), (6, "aBcdef"), (13, "end")], (), _Script((6, "aBcdef"))) == (6, "aBcdef")
        assert _pick_word_for_typo([(0, "start"), (6, "HEllo"), (12, "end")], (), _Script((6, "HEllo"))) == (6, "HEllo")

    def test_a_real_word_is_targeted_even_after_leading_space(self):
        from pyutilz.text.humanizer import _apply_char_typo

        for seed in range(10):
            out, _pos, _delta = _apply_char_typo(" alpha bravo charlie", "skip_letter", (), random.Random(seed))
            parts = out.split()
            assert parts[0] == "alpha" and parts[2] == "charlie"

    def test_no_target_gives_the_text_and_zero_offsets(self):
        from pyutilz.text.humanizer import _apply_char_typo

        assert _apply_char_typo("a b c", "skip_letter", (), random.Random(0)) == ("a b c", 0, 0)

    def test_a_word_of_letters_with_no_keyboard_neighbour_is_left_alone(self):
        from pyutilz.text.humanizer import _apply_char_typo

        assert _apply_char_typo("hi ąęįųū yo", "adjacent_key", (), random.Random(0)) == ("hi ąęįųū yo", 0, 0)

    def test_an_adjacent_key_typo_only_draws_from_letters_that_have_neighbours(self):
        from pyutilz.text.humanizer import _apply_char_typo

        for seed in range(20):
            out, _pos, _delta = _apply_char_typo("start аbcdef end", "adjacent_key", (), random.Random(seed))
            assert out != "start аbcdef end"

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, "tehre"), (2, "three"), (3, "theer")],  # codespell:ignore tehre,theer
    )
    def test_a_transposition_swaps_the_letters_at_the_drawn_inner_position(self, index, expected):
        from pyutilz.text.humanizer import _apply_char_typo

        rec = _Recorder((6, "there"), index)
        out, pos, delta = _apply_char_typo("hello there world", "transpose", (), rec)
        assert out == f"hello {expected} world"
        assert (pos, delta) == (6, 0)
        assert rec.ranges == [(1, 3)]

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, "thhere"), (2, "theere"), (3, "therre")],  # codespell:ignore thhere,theere
    )
    def test_a_double_letter_repeats_the_drawn_inner_letter(self, index, expected):
        from pyutilz.text.humanizer import _apply_char_typo

        rec = _Recorder((6, "there"), index)
        out, pos, delta = _apply_char_typo("hello there world", "double_letter", (), rec)
        assert out == f"hello {expected} world"
        assert (pos, delta) == (6, 1)
        assert rec.ranges == [(1, 3)]

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(1, "tere"), (2, "thre"), (3, "thee")],  # codespell:ignore thre
    )
    def test_a_skipped_letter_is_one_of_the_inner_letters(self, index, expected):
        from pyutilz.text.humanizer import _apply_char_typo

        rec = _Recorder((6, "there"), index)
        out, pos, delta = _apply_char_typo("hello there world", "skip_letter", (), rec)
        assert out == f"hello {expected} world"
        assert (pos, delta) == (6, -1)
        assert rec.ranges == [(1, 3)]

    def test_an_extra_space_is_never_placed_on_a_protected_gap_or_beside_one(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("a b c", "extra_space", ((1, 2),), _Script(3)) == ("a b  c", 3, 1)
        assert _apply_space_typo("a b c", "extra_space", ((4, 5),), _Script(3)) == ("a b  c", 3, 1)

    def test_a_comma_space_is_removed_only_where_it_is_unprotected(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("a, b, c", "missing_space_comma", ((3, 4),), _Script(1)) == ("a,b, c", 2, -1)
        assert _apply_space_typo("a, b, c", "missing_space_comma", ((3, 4),), _Script(4)) == ("a, b,c", 5, -1)

    def test_a_period_space_is_removed_only_where_it_is_unprotected(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("Go. Ok. Yes", "missing_space_period", ((3, 4),), _Script(6)) == ("Go. Ok.Yes", 7, -1)
        assert _apply_space_typo("Go. Ok. Yes", "missing_space_period", ((4, 5),), _Script(2)) == ("Go.Ok. Yes", 3, -1)

    def test_no_period_or_comma_to_hit_leaves_the_text_alone(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("no period here", "missing_space_period", (), random.Random(0)) == ("no period here", 0, 0)
        assert _apply_space_typo("no comma here", "missing_space_comma", (), random.Random(0)) == ("no comma here", 0, 0)

    def test_an_unknown_spacing_typo_leaves_the_text_alone(self):
        from pyutilz.text.humanizer import _apply_space_typo

        assert _apply_space_typo("abc", "bogus", (), random.Random(0)) == ("abc", 0, 0)

    def test_introduce_typos_makes_one_typo_by_default(self):
        from pyutilz.text.humanizer import introduce_typos

        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        for seed in range(10):
            assert introduce_typos(text, rng=random.Random(seed)) == introduce_typos(text, count=1, rng=random.Random(seed))

    def test_a_missing_period_space_chosen_by_the_typo_kind_is_applied(self):
        from pyutilz.text.humanizer import introduce_typos

        assert introduce_typos("Go. Ok.", count=1, rng=_Script("missing_space_period", 2)) == "Go.Ok."

    def test_humanize_makes_one_typo_by_default_and_with_an_explicit_count_of_one(self):
        from pyutilz.text.humanizer import humanize

        assert humanize("Go. Ok.", typo_count=1, rng=_Script("missing_space_period", 2)) == "Go.Ok."

    def test_humanize_default_typo_count_is_one(self):
        from pyutilz.text.humanizer import humanize

        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        for seed in range(10):
            assert humanize(text, rng=random.Random(seed)) == humanize(text, typo_count=1, rng=random.Random(seed))

    def test_a_long_lost_snippet_is_reported_by_its_first_sixty_characters(self, caplog):
        import logging

        from pyutilz.text.humanizer import humanize

        text = "It is worth noting that " + "q" * 70
        with caplog.at_level(logging.WARNING, logger="pyutilz.text.humanizer"):
            humanize(text, typo_count=0, protected_spans=[(0, len(text))])
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(messages) == 1
        assert repr(text[:60]) in messages[0]
        assert repr(text[:61]) not in messages[0]

    def test_a_protected_word_is_never_damaged_by_later_length_changing_typos(self):
        from pyutilz.text.humanizer import introduce_typos

        text = "alpha bravo charlie delta echo foxtrot golf hotel"
        protected = [(31, 38)]
        assert text[31:38] == "foxtrot"
        for seed in range(60):
            out = introduce_typos(text, count=6, rng=random.Random(seed), protected_spans=protected)
            assert "foxtrot" in out, seed

class TestRemainingSurvivorsOfTheSweep:
    """The last survivors the sweep found in the opener removal, the recapitalisation, the proper-noun rule, the play-role
    phrase, the underscore phrase and the seeded typo run (humanizer.py 190, 226, 349, 80, 108, 473)."""

    def test_an_opener_removal_keeps_the_text_before_it(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("Done. Certainly, we ship.") == "Done. We ship."

    def test_a_removed_opener_recapitalises_only_when_the_first_letter_is_lower_case(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("It is worth noting that iPhone works.") == "IPhone works."

    def test_a_capitalised_word_is_the_only_candidate_once_a_proper_noun_is_skipped(self):
        from pyutilz.text.humanizer import _pick_word_for_typo

        rec = _Recorder((12, "WORLD"))
        assert _pick_word_for_typo([(0, "start"), (6, "Hello"), (12, "WORLD"), (18, "end")], (), rec) == (12, "WORLD")
        assert rec.seqs == [[(12, "WORLD")]]

    def test_the_play_role_phrase_needs_its_spacing(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("This playsa vital role here.") == "This playsa vital role here."

    def test_the_underscore_phrase_needs_a_word_boundary_before_it(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("xunderscores its importance here.") == "xunderscores its importance here."

    def test_a_seeded_run_is_reproducible_to_the_character(self):
        import random

        from pyutilz.text.humanizer import introduce_typos

        out = introduce_typos("alpha bravo charlie delta echo foxtrot golf hotel", count=3, rng=random.Random(7))
        assert out == "alpha brao charile delta echo coxtrot golf hotel"  # codespell:ignore brao,charile,coxtrot

class TestProtectedWindowsFollowTheEdits:
    """A protected span that lies after a length-changing typo moves with the text, and a span that ends exactly at the edit
    point is not stretched (humanizer.py 478-484). Checked through the gap positions an extra-space typo may still use."""

    _TEXT = "alpha bravo charlie delta echoes foxtrot end"

    def test_a_protected_span_after_an_edit_moves_with_the_text(self):
        from pyutilz.text.humanizer import introduce_typos

        rec = _Recorder("skip_letter", (6, "bravo"), 2, "extra_space", 18)
        out = introduce_typos(self._TEXT, count=2, rng=rec, protected_spans=((12, 19),))
        assert rec.seqs[1] == [5, 10, 18, 24, 31, 39]
        assert out == "alpha brvo charlie  delta echoes foxtrot end"

    def test_a_span_that_ends_exactly_at_the_edit_point_is_not_stretched(self):
        from pyutilz.text.humanizer import introduce_typos

        rec = _Recorder("skip_letter", (6, "bravo"), 2, "extra_space", 18)
        out = introduce_typos(self._TEXT, count=2, rng=rec, protected_spans=((4, 6),))
        assert rec.seqs[1] == [10, 18, 24, 31, 39]
        assert out == "alpha brvo charlie  delta echoes foxtrot end"

class TestOpenSurvivorsOfTheHumanizerRecheck:
    """Survivors still open after the last re-check: a span that starts at a space after the edited word must move with the
    text on its start alone; the underscore phrase needs its spacing; and an adjacent-key draw only sees letters that have
    neighbours (humanizer.py 80, 379, 483)."""

    def test_a_span_starting_after_a_space_moves_its_start_with_the_edit(self):
        from pyutilz.text.humanizer import introduce_typos

        rec = _Recorder("skip_letter", (6, "bravo"), 2, "extra_space", 18)
        out = introduce_typos("alpha bravo charlie delta echoes foxtrot end", count=2, rng=rec, protected_spans=((11, 19),))
        assert rec.seqs[1] == [5, 18, 24, 31, 39]
        assert out == "alpha brvo charlie  delta echoes foxtrot end"

    def test_the_underscore_phrase_needs_its_spacing_after_the_verb(self):
        from pyutilz.text.humanizer import strip_ai_patterns

        assert strip_ai_patterns("This underscoresits importance here.") == "This underscoresits importance here."

    def test_an_adjacent_key_draw_only_considers_letters_with_neighbours(self):
        from pyutilz.text.humanizer import _apply_char_typo

        rec = _Recorder((6, "ąbcdef"), 1, "v")
        out, pos, delta = _apply_char_typo("start ąbcdef end", "adjacent_key", (), rec)
        assert rec.seqs[1] == [1, 2, 3, 4, 5]
        assert out == "start ąvcdef end"
        assert (pos, delta) == (6, 0)

class TestHumanizeSpanRemappingExactly:
    """What humanize hands to introduce_typos: the cursor starts at zero, a repeated snippet claims its next occurrence,
    a snippet found only through the global fallback keeps its position, and nothing is lost when the cursor search fails
    but the fallback succeeds (humanizer.py 540-556)."""

    def _capture(self, monkeypatch):
        from pyutilz.text import humanizer as h

        seen: list = []

        def _fake(text, count=1, rng=None, protected_spans=()):
            seen.append(list(protected_spans))
            return text

        monkeypatch.setattr(h, "introduce_typos", _fake)
        return h, seen

    def test_the_first_occurrence_is_claimed_from_the_start_of_the_text(self, monkeypatch):
        h, seen = self._capture(monkeypatch)
        h.humanize("aa", typo_count=1, protected_spans=[(0, 1)])
        assert seen == [[(0, 1)]]

    def test_a_repeated_snippet_claims_its_next_occurrence_after_the_first(self, monkeypatch):
        h, seen = self._capture(monkeypatch)
        h.humanize("aaa", typo_count=1, protected_spans=[(0, 1), (0, 1)])
        assert seen == [[(0, 1), (1, 2)]]

    def test_a_snippet_found_only_by_the_fallback_keeps_its_earlier_position(self, monkeypatch):
        h, seen = self._capture(monkeypatch)
        import logging

        records: list = []
        handler = logging.Handler()
        handler.emit = records.append
        logging.getLogger("pyutilz.text.humanizer").addHandler(handler)
        try:
            h.humanize("ab", typo_count=1, protected_spans=[(0, 1), (0, 2)])
        finally:
            logging.getLogger("pyutilz.text.humanizer").removeHandler(handler)
        assert seen == [[(0, 1), (0, 2)]]
        assert not [r for r in records if r.levelno >= logging.WARNING]
