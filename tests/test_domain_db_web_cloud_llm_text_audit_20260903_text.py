"""Behavioural regression tests for the text-domain findings of the 2026-09-03 audit
(``audits/2026-09-03/09-domain-db-web-cloud-llm-text.md``).

One test (or small group) per finding: F05, F06, F07, F26, F27, F28, F29, F30, F57-F63, F64.
Every assertion is behavioural -- no source inspection.
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from pyutilz.text.humanizer import introduce_typos, strip_ai_patterns, strip_emojis
from pyutilz.text.secrets_scrub import redact_secrets
from pyutilz.text.similarity import sentences_similarity
from pyutilz.text.strings.basics import find_between
from pyutilz.text.strings.configfiles import read_config_file, write_config_file
from pyutilz.text.strings.jsonutils import json_pg_dumps
from pyutilz.text.strings.textentropy import compute_entropy_stats, naive_entropy_rate
from pyutilz.text.strings.webtext import fix_broken_sentences, fix_missed_space_between_sentences

# ---------------------------------------------------------------------------
# F05 -- fix_broken_sentences IndexError on "<linebreak><space>" at end of text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["\n ", "\r\n ", "Hello world.\nA b\n "])
def test_f05_fix_broken_sentences_no_indexerror_on_trailing_linebreak_space(text: str) -> None:
    """`j + 1 <= text_len` read one past the end; any text ending in linebreak+space crashed."""
    assert isinstance(fix_broken_sentences(text), str)


# ---------------------------------------------------------------------------
# F58 -- backward scan started at -1 and read the LAST character of the string
# ---------------------------------------------------------------------------


def test_f58_leading_linebreak_result_does_not_depend_on_the_final_character() -> None:
    """With the match at index 0 the "previous symbol" scan used to wrap to ``text[-1]``.

    The two inputs below differ ONLY in their last character, which cannot be the character
    preceding a linebreak at index 0; before the fix ``"\\nAbc."`` became ``" Abc."`` (the
    trailing dot, read as the "previous" symbol, took the eos branch) while ``"\\nAbcd"`` did not.
    """
    with_eos_tail = fix_broken_sentences("\nAbc.")
    without_eos_tail = fix_broken_sentences("\nAbcd.")
    assert with_eos_tail.startswith("\n"), repr(with_eos_tail)
    assert without_eos_tail.startswith("\n"), repr(without_eos_tail)


# ---------------------------------------------------------------------------
# F06 -- numba / pure-Python parity on combining marks
# ---------------------------------------------------------------------------

_NFD_WORD = "café"  # "cafe" + COMBINING ACUTE ACCENT
_NFC_WORD = "café"  # precomposed


def test_f06_pure_python_similarity_is_nfc_normalized() -> None:
    assert unicodedata.normalize("NFC", _NFD_WORD) == _NFC_WORD
    assert sentences_similarity([_NFD_WORD], [_NFC_WORD]) == 1.0


def test_f06_numba_and_pure_python_agree_on_nfd_vs_nfc() -> None:
    pytest.importorskip("numba")
    from pyutilz.text.similarity import sentences_similarity_numba

    pure = sentences_similarity([_NFD_WORD], [_NFC_WORD])
    fast = sentences_similarity_numba([_NFD_WORD], [_NFC_WORD])
    assert pure == pytest.approx(fast), (pure, fast)

    pure2 = sentences_similarity([_NFD_WORD, "madrid"], [_NFC_WORD, "madridd"])
    fast2 = sentences_similarity_numba([_NFD_WORD, "madrid"], [_NFC_WORD, "madridd"])
    assert pure2 == pytest.approx(fast2), (pure2, fast2)


# ---------------------------------------------------------------------------
# F07 -- redact_secrets false negatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "secret_key=abc123",
        "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG",
        "api_key_id=zzz",
        "MY_AUTH_TOKEN_V2: hunter2hunter2",
    ],
)
def test_f07_key_name_affixes_are_redacted(text: str) -> None:
    """A key that merely CONTAINS a secret keyword must redact too, not only one that ends in it."""
    out = redact_secrets(text)
    assert out.endswith("=***"), repr(out)


@pytest.mark.parametrize(
    "text",
    [
        "ghp_1234567890abcdefghijklmnopqrstuvwxyzAB",
        "AKIAIOSFODNN7EXAMPLE",
        "sk-proj-ABCDEFGHIJKLMNOPQRSTUV",
    ],
)
def test_f07_standalone_high_entropy_tokens_are_redacted(text: str) -> None:
    out = redact_secrets("leaked " + text + " here")
    assert text not in out, repr(out)
    assert "***" in out, repr(out)


def test_f07_pem_private_key_block_is_redacted() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\nsecretbody\n-----END RSA PRIVATE KEY-----"
    out = redact_secrets("dump:\n" + pem + "\ntail")
    assert "secretbody" not in out, repr(out)
    assert "BEGIN RSA PRIVATE KEY" not in out, repr(out)
    assert out.endswith("tail"), repr(out)


def test_f07_high_entropy_patterns_do_not_backtrack_catastrophically() -> None:
    """A long non-matching run must stay linear (each alternative is prefix + one bounded class)."""
    import time

    payload = "sk-" + "A" * 20000 + " "
    t0 = time.perf_counter()
    redact_secrets(payload + "x" * 20000)
    assert time.perf_counter() - t0 < 2.0


def test_f07_ordinary_prose_still_not_redacted() -> None:
    assert redact_secrets("and oauth flows here") == "and oauth flows here"


# ---------------------------------------------------------------------------
# F57 -- redact_secrets must not match across a newline
# ---------------------------------------------------------------------------


def test_f57_redaction_does_not_span_a_line_break() -> None:
    text = "Missing password:\nTraceback (most recent call last):"
    out = redact_secrets(text)
    assert "Traceback" in out, repr(out)
    assert "\n" in out, repr(out)


def test_f57_same_line_secret_still_redacted() -> None:
    out = redact_secrets("password: hunter2\nTraceback (most recent call last):")
    assert "hunter2" not in out, repr(out)
    assert "Traceback" in out, repr(out)


# ---------------------------------------------------------------------------
# F26 -- config round-trip must not double "%"
# ---------------------------------------------------------------------------


def test_f26_percent_survives_repeated_config_round_trips(tmp_path) -> None:
    path = str(tmp_path / "cfg.ini")
    value = "50% off"
    for _ in range(3):
        assert write_config_file(path, {"rate": value}, section="MAIN", encryption=None) is True
        out: dict = {}
        assert read_config_file(path, out, section="MAIN", variables=["rate"], encryption=None) is True
        assert out["rate"] == "50% off", repr(out["rate"])
        value = out["rate"]


# ---------------------------------------------------------------------------
# F59 -- read_config_file failure branch must use the same (prefixed) key shape
# ---------------------------------------------------------------------------


def test_f59_missing_variable_uses_the_prefixed_key_in_all_sections_mode(tmp_path) -> None:
    path = tmp_path / "two.ini"
    path.write_text("[S1]\na = 1\n\n[S2]\na = 2\n", encoding="utf-8")
    out: dict = {}
    assert read_config_file(str(path), out, section=None, variables=["a", "missing"], encryption=None) is True
    assert out["s1_a"] == 1
    assert out["s2_a"] == 2
    assert out["s1_missing"] is None
    assert out["s2_missing"] is None
    assert "missing" not in out, out


# ---------------------------------------------------------------------------
# F27 -- introduce_typos must keep protected spans valid across length changes
# ---------------------------------------------------------------------------


def _typo_fixture_text() -> "tuple[str, tuple[int, int]]":
    prefix = " ".join(["word%d," % i for i in range(60)]) + " "
    protected_word = "abcde"
    suffix = " " + " ".join(["tail%d." % i for i in range(10)])
    text = prefix + protected_word + suffix
    return text, (len(prefix), len(prefix) + len(protected_word))


def test_f27_protected_span_survives_many_length_changing_typos() -> None:
    text, span = _typo_fixture_text()
    corrupted = []
    for seed in range(500):
        out = introduce_typos(text, count=40, rng=random.Random(seed), protected_spans=[span])
        if out.count("abcde") != 1:
            corrupted.append(seed)
    assert corrupted == [], "protected span corrupted for seeds %r" % corrupted[:10]


# ---------------------------------------------------------------------------
# F28 -- fix_missed_space_between_sentences must not split decimals
# ---------------------------------------------------------------------------


def test_f28_decimal_numbers_are_not_split() -> None:
    assert fix_missed_space_between_sentences("Version 3.5 costs $2.50") == "Version 3.5 costs $2.50"
    assert fix_missed_space_between_sentences("addr 192.168.0.1 ok") == "addr 192.168.0.1 ok"


def test_f28_real_sentence_boundaries_still_get_a_space() -> None:
    assert fix_missed_space_between_sentences("Hello.World") == "Hello. World"
    assert fix_missed_space_between_sentences("Done.42 items") == "Done. 42 items"


# ---------------------------------------------------------------------------
# F29 -- sentencize_text must strip only the LEADING/TRAILING separator
# ---------------------------------------------------------------------------


@pytest.fixture()
def _no_emoji_tables(monkeypatch):
    """`sentencize_text` lazily loads emoji tables from the optional ``emoji_data_python``.

    The tilda handling under test runs before that, so an empty table keeps the test offline.
    """
    from pyutilz.text.strings import webtext

    monkeypatch.setattr(webtext, "ascii_emojies", set(), raising=False)
    monkeypatch.setattr(webtext, "unicode_emojies", set(), raising=False)
    return webtext


def test_f29_interior_separators_survive_leading_or_trailing_stripping(_no_emoji_tables) -> None:
    st = _no_emoji_tables.sentencize_text
    assert st("- Great product - really") == "Great product - really."
    assert st("Hello - world -") == "Hello - world."
    assert st("~ x ~ y ~") == "X ~ y."


# ---------------------------------------------------------------------------
# F30 -- json_pg_dumps must not depend on whether orjson is installed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({1: "intkey"}, '{"1": "intkey"}'),
        ({"s": "\ud800"}, '{"s": "\\ud800"}'),
    ],
)
def test_f30_json_pg_dumps_handles_shapes_orjson_rejects(obj: dict, expected: str) -> None:
    # ``dumps()`` returns the finished JSON text; ``__str__`` would wrap it in an SQL literal and
    # re-escape the backslashes, which is not what this test is about.
    assert json_pg_dumps(obj).dumps(None) == expected  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# F60 -- find_between window bounds
# ---------------------------------------------------------------------------


def test_f60_empty_end_honours_idx2() -> None:
    assert find_between("abcXdefYghi", "X", "", 0, 8) == "defY"


def test_f60_negative_idx1_with_empty_start_uses_python_slice_semantics() -> None:
    assert find_between("abcdef", "", "d", -3) == ""
    assert find_between("abcdefdz", "", "d", -3) == "f"


# ---------------------------------------------------------------------------
# F61 -- entropy helpers must never return -0.0
# ---------------------------------------------------------------------------


def test_f61_zero_entropy_is_positive_zero() -> None:
    raw, rate = compute_entropy_stats("a")
    assert repr(raw) == "0.0", repr(raw)
    assert repr(naive_entropy_rate("aaaa")) == "0.0", repr(naive_entropy_rate("aaaa"))
    assert rate == 0.0


def test_f61_nonzero_entropy_unchanged() -> None:
    assert naive_entropy_rate("ab") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F62 -- strip_emojis: keep check marks, remove flags
# ---------------------------------------------------------------------------


def test_f62_check_marks_are_kept() -> None:
    text = "check ✓ ok ✔"
    assert strip_emojis(text) == text, ascii(strip_emojis(text))


def test_f62_flag_emoji_are_removed() -> None:
    out = strip_emojis("hi \U0001f1fa\U0001f1f8 there")
    assert "\U0001f1fa" not in out, ascii(out)
    assert "\U0001f1f8" not in out, ascii(out)


def test_f62_ordinary_emoji_still_removed() -> None:
    assert "\U0001f600" not in strip_emojis("hey \U0001f600")
    assert "⭐" not in strip_emojis("star ⭐")


# ---------------------------------------------------------------------------
# F63 -- single-word sentences count as last-word too
# ---------------------------------------------------------------------------


def test_f63_single_word_sentence_counts_as_last_word() -> None:
    pytest.importorskip("nltk")
    from pyutilz.text.tokenizers import AdvancedTokenizer

    tok = AdvancedTokenizer()
    tok.tokenize("Hello")
    assert tok.NUM_FIRSTWORD_INSENTENCE["hello"] == 1
    assert tok.NUM_LASTWORD_INSENTENCE["hello"] == 1
# ---------------------------------------------------------------------------
# F64 -- strip_ai_patterns removes hedging openers and parenthetical justifications
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, gone",
    [
        ("Certainly! Here is the code.", "Certainly"),
        ("Sure thing, I can do that.", "Sure thing"),
        ("Great question! The answer is 42.", "Great question"),
        ("Absolutely, that works.", "Absolutely"),
        ("Of course. The file is missing.", "Of course"),
        ("I'd be happy to help! Run the script.", "happy to help"),
        ("Done.\nCertainly! Now restart.", "Certainly"),
    ],
)
def test_f64_hedging_openers_are_stripped(text: str, gone: str) -> None:
    out = strip_ai_patterns(text)
    assert gone.lower() not in out.lower()
    assert out
    assert out[0].isupper()


@pytest.mark.parametrize(
    "text",
    [
        "We are certainly not done yet.",
        "The result is absolutely wrong.",
        "He knew, of course, that it would fail.",
    ],
)
def test_f64_the_same_words_survive_mid_sentence(text: str) -> None:
    """Only the OPENER slot is a hedge; mid-sentence the words are ordinary English."""
    assert strip_ai_patterns(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Use a dict comprehension (more idiomatic) instead.", "Use a dict comprehension instead."),
        ("Prefer pathlib (which is cleaner) here.", "Prefer pathlib here."),
        ("Return early (the standard approach).", "Return early."),
    ],
)
def test_f64_parenthetical_self_justifications_are_stripped(text: str, expected: str) -> None:
    assert strip_ai_patterns(text) == expected


def test_f64_ordinary_parentheticals_survive() -> None:
    """A closed vocabulary, matched only when it is the WHOLE parenthetical."""
    text = "Install it (the wheel ships in 3.12) before running."
    assert strip_ai_patterns(text) == text


def test_f64_lowercase_input_is_not_recapitalised() -> None:
    assert strip_ai_patterns("certainly! here is the code.") == "here is the code."
