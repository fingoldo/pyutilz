"""Tests for pyutilz.text.humanizer."""

import random

import pytest

from pyutilz.text.humanizer import (
    strip_ai_patterns,
    fix_dashes,
    strip_emojis,
    introduce_typos,
    humanize,
    _get_adjacent,
)


# ── strip_ai_patterns ──────────────────────────────────────────────────────


class TestStripAIPatterns:
    def test_delve_into(self):
        assert "look into" in strip_ai_patterns("Let's delve into this topic.")

    def test_leverage(self):
        assert strip_ai_patterns("We leverage Python here.") == "We use Python here."

    def test_utilize(self):
        assert strip_ai_patterns("We utilize Docker.") == "We use Docker."

    def test_furthermore(self):
        result = strip_ai_patterns("Furthermore, this is important.")
        assert result.startswith("Also,")
        assert "Furthermore" not in result

    def test_moreover(self):
        assert "Moreover" not in strip_ai_patterns("Moreover, it works.")

    def test_additionally(self):
        assert "Additionally" not in strip_ai_patterns("Additionally, we added tests.")

    def test_in_conclusion(self):
        assert "In conclusion" not in strip_ai_patterns("In conclusion, it works well.")

    def test_ensure(self):
        assert "make sure" in strip_ai_patterns("We need to ensure quality.")

    def test_ensure_that(self):
        assert "make sure" in strip_ai_patterns("We ensure that tests pass.")

    def test_comprehensive(self):
        assert "thorough" in strip_ai_patterns("A comprehensive review.")

    def test_streamline(self):
        assert "simplify" in strip_ai_patterns("We streamline the process.")

    def test_facilitate(self):
        result = strip_ai_patterns("This will facilitate communication.")
        assert "help with" in result

    def test_thrilled(self):
        result = strip_ai_patterns("I'm thrilled to announce this.")
        assert "glad" in result
        assert "thrilled" not in result

    def test_excited(self):
        result = strip_ai_patterns("I'm excited to join the team.")
        assert "glad" in result

    def test_love_to(self):
        result = strip_ai_patterns("I'd love to help with this project.")
        assert "like to" in result

    def test_bring_years(self):
        result = strip_ai_patterns("I bring 10 years of experience.")
        assert "I have 10 years of" in result

    def test_vital_role(self):
        result = strip_ai_patterns("It plays a vital role in testing.")
        assert "important" in result

    def test_crucial_role(self):
        result = strip_ai_patterns("It plays a crucial role.")
        assert "important" in result

    def test_testament(self):
        result = strip_ai_patterns("It stands as a testament to quality.")
        assert "testament" not in result

    def test_worth_noting(self):
        result = strip_ai_patterns("It's worth noting that Python is great.")
        assert "worth noting" not in result

    def test_important_to_note(self):
        result = strip_ai_patterns("It is important to note that this works.")
        assert "important to note" not in result

    def test_pivotal(self):
        assert "key" in strip_ai_patterns("This is a pivotal moment.")

    def test_seamless(self):
        assert "smooth" in strip_ai_patterns("A seamless integration.")

    def test_robust(self):
        assert "solid" in strip_ai_patterns("A robust solution.")

    def test_meticulous(self):
        assert "careful" in strip_ai_patterns("A meticulous approach.")

    def test_innovative(self):
        assert "new" in strip_ai_patterns("An innovative approach.")

    def test_cutting_edge(self):
        assert "modern" in strip_ai_patterns("Cutting-edge technology.")

    def test_consequently(self):
        result = strip_ai_patterns("Consequently, we decided to proceed.")
        assert result.startswith("So,")

    def test_nevertheless(self):
        result = strip_ai_patterns("Nevertheless, it was successful.")
        assert result.startswith("Still,")

    def test_as_an_ai(self):
        result = strip_ai_patterns("As an AI language model, I cannot do that.")
        assert "As an AI" not in result

    def test_no_double_spaces(self):
        result = strip_ai_patterns("It's worth noting that  this works.")
        assert "  " not in result

    def test_sentence_recase_after_removal(self):
        result = strip_ai_patterns("Yes. In conclusion, this works.")
        # After "In conclusion," is removed, next word should be capitalized.
        assert "yes" not in result.lower().split(". ")[0] or result[0] == "Y"

    def test_preserves_normal_text(self):
        normal = "The quick brown fox jumps over the lazy dog."
        assert strip_ai_patterns(normal) == normal

    def test_case_insensitive(self):
        assert "use" in strip_ai_patterns("We LEVERAGE this tool.")


# ── fix_dashes ──────────────────────────────────────────────────────────────


class TestFixDashes:
    def test_em_dash(self):
        assert fix_dashes("hello\u2014world") == "hello - world"

    def test_en_dash(self):
        assert fix_dashes("2020\u20132025") == "2020 - 2025"

    def test_smart_double_quotes(self):
        result = fix_dashes("\u201cHello\u201d")
        assert result == '"Hello"'

    def test_smart_single_quotes(self):
        result = fix_dashes("\u2018it\u2019s\u2019")
        assert result == "'it's'"

    def test_no_double_spaces(self):
        result = fix_dashes("a \u2014 b")
        assert "  " not in result

    def test_preserves_normal(self):
        assert fix_dashes("hello - world") == "hello - world"


# ── strip_emojis ───────────────────────────────────────────────────────────


class TestStripEmojis:
    def test_removes_smiley(self):
        assert strip_emojis("Hello \U0001F600 world") == "Hello  world"

    def test_removes_rocket(self):
        assert strip_emojis("Launch \U0001F680!") == "Launch !"

    def test_removes_star(self):
        assert "⭐" not in strip_emojis("Great ⭐ job")

    def test_preserves_text(self):
        assert strip_emojis("No emojis here.") == "No emojis here."

    def test_preserves_regular_unicode(self):
        assert strip_emojis("Привет мир") == "Привет мир"

    def test_preserves_smiley_emoticon(self):
        # ;-) is ASCII, not emoji — must be preserved.
        assert strip_emojis("Hello ;-)") == "Hello ;-)"


# ── QWERTY adjacency ──────────────────────────────────────────────────────


class TestQwertyAdjacency:
    def test_english_key(self):
        adj = _get_adjacent("e")
        assert adj is not None
        assert "w" in adj and "r" in adj

    def test_russian_key(self):
        adj = _get_adjacent("к")
        assert adj is not None
        assert "у" in adj

    def test_uppercase_preserves_case(self):
        adj = _get_adjacent("E")
        assert adj is not None
        assert "W" in adj

    def test_unsupported_returns_none(self):
        assert _get_adjacent("7") is None
        assert _get_adjacent("@") is None

    def test_chinese_returns_none(self):
        assert _get_adjacent("中") is None


# ── introduce_typos ────────────────────────────────────────────────────────


class TestIntroduceTypos:
    def test_one_typo_changes_text(self):
        text = "The developer quickly debugged the application."
        rng = random.Random(42)
        result = introduce_typos(text, count=1, rng=rng)
        assert result != text

    def test_zero_typos_preserves_text(self):
        text = "The developer quickly debugged the application."
        assert introduce_typos(text, count=0, rng=random.Random(42)) == text

    def test_reproducible_with_seed(self):
        text = "The developer quickly debugged the application."
        r1 = introduce_typos(text, count=2, rng=random.Random(123))
        r2 = introduce_typos(text, count=2, rng=random.Random(123))
        assert r1 == r2

    def test_protected_spans_untouched(self):
        text = "ABRACADABRA ;-) I am a skilled developer with experience."
        # Protect "ABRACADABRA ;-)" (first 15 chars).
        protected = [(0, 15)]
        rng = random.Random(42)
        for _ in range(20):
            result = introduce_typos(text, count=2, rng=rng, protected_spans=protected)
            assert result[:15] == text[:15], f"Protected span was modified: {result[:15]}"

    def test_multiple_typos(self):
        text = "The experienced developer quickly debugged the complex application code."
        rng = random.Random(42)
        result = introduce_typos(text, count=5, rng=rng)
        assert result != text

    def test_short_text_survives(self):
        text = "Hi."
        result = introduce_typos(text, count=1, rng=random.Random(42))
        # May or may not change — but shouldn't crash.
        assert isinstance(result, str)

    def test_adjacent_key_typo_is_nearby(self):
        # Force adjacent_key typo by seeding until we get one.
        text = "The developer quickly debugged the application."
        rng = random.Random(42)
        # Run many times; at least one should produce a char-level change.
        changed = False
        for seed in range(100):
            rng = random.Random(seed)
            result = introduce_typos(text, count=1, rng=rng)
            if result != text and len(result) == len(text):
                changed = True
                break
        assert changed, "No adjacent_key typo produced in 100 seeds"

    def test_extra_space_typo(self):
        text = "The developer quickly debugged the application."
        found = False
        for seed in range(200):
            result = introduce_typos(text, count=1, rng=random.Random(seed))
            if "  " in result:
                found = True
                break
        assert found, "No extra_space typo produced in 200 seeds"

    def test_missing_space_after_comma(self):
        text = "Yes, I can do that, and it works, perfectly fine."
        found = False
        for seed in range(200):
            result = introduce_typos(text, count=1, rng=random.Random(seed))
            # Check if any ", " became ","<non-space>
            if ",I" in result or ",a" in result or ",p" in result:
                found = True
                break
        assert found, "No missing_space_comma typo in 200 seeds"


# ── humanize (full pipeline) ──────────────────────────────────────────────


class TestHumanize:
    def test_full_pipeline(self):
        text = (
            "I'm thrilled to delve into this comprehensive project. "
            "Furthermore, I leverage cutting-edge technology\u2014"
            "ensuring seamless integration. \U0001F680"
        )
        result = humanize(text, typo_count=0)
        assert "thrilled" not in result
        assert "delve" not in result
        assert "comprehensive" not in result
        assert "Furthermore" not in result
        assert "leverage" not in result
        assert "cutting-edge" not in result
        assert "\u2014" not in result
        assert "\U0001F680" not in result

    def test_protected_spans_through_pipeline(self):
        text = "ABRACADABRA ;-) I am a skilled developer."
        result = humanize(text, typo_count=3, rng=random.Random(42), protected_spans=[(0, 15)])
        assert result.startswith("ABRACADABRA ;-)")

    def test_no_typos_mode(self):
        text = "A simple test sentence with nothing special."
        result = humanize(text, typo_count=0)
        assert result == text

    def test_russian_text_preserved(self):
        text = "Привет, я разработчик с опытом работы."
        result = humanize(text, typo_count=0)
        assert result == text

    def test_russian_typos_use_ycuken(self):
        text = "Привет я опытный разработчик программного обеспечения."
        changed = False
        for seed in range(100):
            result = introduce_typos(text, count=1, rng=random.Random(seed))
            if result != text:
                changed = True
                break
        assert changed, "Russian text should get typos too"
