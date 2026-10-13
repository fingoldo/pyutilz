"""Continuation of test_domain_db_web_cloud_llm_text_audit_20260902: split out to keep each test module under the 1k-LOC gate."""
import asyncio
from pyutilz.dev.logginglib import reset_log_throttles
import sys
import threading
import pytest
from pyutilz.llm.exceptions import JSONParsingError
import pyutilz.llm.claude_code_provider as ccmod
from pyutilz.text.similarity import normalize_sentence, sentences_similarity
from pyutilz.text.strings.textentropy import compute_entropy_stats
from pyutilz.text.secrets_scrub import redact_secrets, sanitize_dsn
from pyutilz.text.strings.jsonutils import json_pg_dumps
from pyutilz.text.strings.webtext import remove_videos, unescape_html
from pyutilz.text.strings.basics import find_between
from pyutilz.text.humanizer import fix_dashes

def _make_openai_compat_provider():
    """A minimal concrete subclass: the base class is abstract on the two pricing hooks."""
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider

    class _Stub(OpenAICompatibleProvider):
        def _input_cost_per_1m(self, model: str) -> float:
            return 0.0

        def _output_cost_per_1m(self, model: str) -> float:
            return 0.0

    p = _Stub.__new__(_Stub)
    p.model_name = "test-model"
    return p


_BS = chr(92)


def _require_spacy_model(name: str = "en_core_web_sm") -> None:
    """Skips unless the spaCy model AdvancedTokenizer loads in __init__ is present.

    spaCy installs the package but never the models -- those are a separate download, so the
    precondition is an explicit skip naming what is missing rather than a silently passing test.
    """
    spacy = pytest.importorskip("spacy")
    # is_package is a pure existence check -- loading the model inside a try/except whose handler
    # skips would let a genuine breakage in the model-loading path masquerade as "not installed".
    if not spacy.util.is_package(name):
        pytest.skip(f"spaCy model {name!r} is not downloaded on this host")


def test_f53_base_reset_per_call_state_clears_stale_metadata():
    """The base implementation was an empty no-op, so for OpenAI/xAI/DeepSeek a generate() that
    raised left the previous call's usage/tool_calls readable as the latest one."""
    p = _make_openai_compat_provider()
    p.last_tool_calls = [{"name": "x"}]
    p.last_citations = ["c"]
    p._last_finish_reason = "stop"
    p._last_usage = {"input_tokens": 5, "output_tokens": 7, "reasoning_tokens": 0}
    p._reset_per_call_state()
    assert p.last_tool_calls == []
    assert p.last_citations == []
    assert p._last_finish_reason is None
    assert p._last_usage["input_tokens"] == 0


# ---------------------------------------------------------------------------
# F08 / F57 -- OpenRouter per-call metadata
# ---------------------------------------------------------------------------


def test_f08_openrouter_per_call_metadata_is_context_scoped():
    """These were plain instance attributes, so generate_batch's N concurrent calls zeroed each
    other's in-flight cost and generation id -- a cost-accounting loop billed the wrong request."""
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    p = OpenRouterProvider.__new__(OpenRouterProvider)
    results = {}

    async def call(tag, cost, gen_id):
        p.last_actual_cost_usd = cost
        p.last_generation_id = gen_id
        await asyncio.sleep(0)
        results[tag] = (p.last_actual_cost_usd, p.last_generation_id)

    async def main():
        await asyncio.gather(
            asyncio.create_task(call("a", 0.42, "gen-a")),
            asyncio.create_task(call("b", 0.01, "gen-b")),
        )

    asyncio.run(main())
    assert results["a"] == (0.42, "gen-a")
    assert results["b"] == (0.01, "gen-b")


def test_f08_metadata_names_are_carried_in_the_batch_snapshot():
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    attrs = OpenRouterProvider._PERCALL_METADATA_ATTRS
    for name in ("last_actual_cost_usd", "last_generation_id", "last_upstream_provider", "last_web_search_citations"):
        assert name in attrs


# ---------------------------------------------------------------------------
# F11 -- factory LRU eviction
# ---------------------------------------------------------------------------


def test_f11_evicting_a_still_referenced_provider_does_not_close_it(monkeypatch):
    """Eviction was keyed purely on cache size, not liveness: get_llm_provider() hands out shared
    instances and keeps no refcount, so `_close()` could fire on a provider a coroutine obtained
    earlier and was still using ("Cannot send a request, as the client has been closed")."""
    import types

    import pyutilz.llm.factory as factorymod

    closed = []

    class _FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def _close(self):
            closed.append(self)

    fake_mod = types.ModuleType("pyutilz_test_fake_provider_module")
    fake_mod.FakeProvider = _FakeProvider
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)
    monkeypatch.setitem(factorymod._PROVIDER_MODULES, "fake", (fake_mod.__name__, "FakeProvider", None))
    monkeypatch.setattr(factorymod, "_PROVIDER_CACHE_MAX_SIZE", 1)
    monkeypatch.setattr(factorymod, "_schedule_provider_close", lambda p: closed.append(p))
    factorymod._provider_cache.clear()

    held = factorymod.get_llm_provider("fake", model="a")
    factorymod.get_llm_provider("fake", model="b")  # evicts the first entry
    assert closed == [], "a provider the caller still holds must not be closed on eviction"
    assert held.kwargs["model"] == "a"

    factorymod._provider_cache.clear()


def test_f11_evicting_an_unreferenced_provider_still_closes_it(monkeypatch):
    import types

    import pyutilz.llm.factory as factorymod

    closed = []

    class _FakeProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_mod = types.ModuleType("pyutilz_test_fake_provider_module2")
    fake_mod.FakeProvider = _FakeProvider
    monkeypatch.setitem(sys.modules, fake_mod.__name__, fake_mod)
    monkeypatch.setitem(factorymod._PROVIDER_MODULES, "fake2", (fake_mod.__name__, "FakeProvider", None))
    monkeypatch.setattr(factorymod, "_PROVIDER_CACHE_MAX_SIZE", 1)
    monkeypatch.setattr(factorymod, "_schedule_provider_close", lambda p: closed.append(p))
    factorymod._provider_cache.clear()

    factorymod.get_llm_provider("fake2", model="a")  # no reference kept
    factorymod.get_llm_provider("fake2", model="b")
    assert len(closed) == 1

    factorymod._provider_cache.clear()


# ---------------------------------------------------------------------------
# F26 / F32 / F33 / F56 -- ClaudeCodeProvider
# ---------------------------------------------------------------------------

def test_f56_impossible_reset_date_returns_none_not_valueerror():
    """The ValueError escaped from INSIDE the rate-limit handler, turning a recoverable pause
    into a hard failure of generate()."""
    assert ccmod._parse_reset_wait_seconds("Your limit resets Feb 30, 3pm") is None


def test_f56_valid_reset_date_still_parses():
    assert ccmod._parse_reset_wait_seconds("Your limit resets Mar 3, 11pm") > 0


def test_f33_generate_json_uses_the_shared_parser():
    """It re-implemented parsing with a greedy pattern spanning the first { to the LAST }, never
    ran the refusal check, and raised a bare ValueError instead of the typed exceptions."""
    p = ccmod.ClaudeCodeProvider.__new__(ccmod.ClaudeCodeProvider)

    async def fake_generate(**kw):
        return 'prose {"a": 1} more prose {"b": 2}'

    p.generate = fake_generate
    assert asyncio.run(p.generate_json("x")) == {"a": 1}


def test_f33_unparseable_json_raises_the_typed_error():
    p = ccmod.ClaudeCodeProvider.__new__(ccmod.ClaudeCodeProvider)

    async def fake_generate(**kw):
        return "not json at all"

    p.generate = fake_generate
    with pytest.raises(JSONParsingError):
        asyncio.run(p.generate_json("x"))


# ---------------------------------------------------------------------------
# F30 / F31 -- Anthropic / Gemini session cost
# ---------------------------------------------------------------------------


def test_f30_anthropic_session_cost_reports_total_cost_usd():
    """Provider-agnostic spend reporting reads this key; Anthropic alone raised KeyError."""
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = "claude-sonnet-4-5"
    p._call_count = 2
    p.total_input_tokens = 1000
    p.total_output_tokens = 500
    p.total_thinking_tokens = 0
    p.total_cache_creation_input_tokens = 0
    p.total_cache_read_input_tokens = 0
    cost = p.get_session_cost()
    assert cost["total_cost_usd"] == pytest.approx(cost["input_cost_usd"] + cost["output_cost_usd"])


def test_f30_gemini_has_a_session_cost():
    """GeminiProvider defined none at all, so a multi-provider dashboard hit AttributeError."""
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    p._call_count = 1
    p.total_prompt_tokens = 1_000_000
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    p.total_cached_content_tokens = 0
    assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.30)


def test_f30_gemini_cache_hits_are_billed_at_the_cache_rate():
    """_CACHE_HIT_COST was unreferenced dead code."""
    from pyutilz.llm.gemini_provider import GeminiProvider

    p = GeminiProvider.__new__(GeminiProvider)
    p.model_name = "gemini-2.5-flash"
    p._call_count = 1
    p.total_prompt_tokens = 1_000_000
    p.total_completion_tokens = 0
    p.total_reasoning_tokens = 0
    p.total_cached_content_tokens = 1_000_000
    assert p.get_session_cost()["total_cost_usd"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# F36 / F54 -- pricing / limit tables
# ---------------------------------------------------------------------------


def test_f36_every_priced_openai_model_has_the_other_table_entries():
    from pyutilz.llm.openai_provider import _CACHE_HIT_COST, _CONTEXT_WINDOW, _MAX_TOKENS, _PRICING

    assert not set(_PRICING) - set(_CACHE_HIT_COST)
    assert not set(_PRICING) - set(_MAX_TOKENS)
    assert not set(_PRICING) - set(_CONTEXT_WINDOW)


def test_f36_o1_pro_cache_rate_is_not_borrowed_from_gpt5_mini():
    from pyutilz.llm.openai_provider import _CACHE_HIT_COST, _PRICING

    assert _CACHE_HIT_COST["o1-pro"] > _PRICING["gpt-5-mini"][0]


def test_f36_dated_snapshot_inherits_its_family_budget():
    """Exact dict.get sent any dated snapshot id to the 16,384 / 128,000 class defaults."""
    from pyutilz.llm.openai_provider import OpenAIProvider

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model_name = "gpt-5-mini-2026-03-01"
    assert p.max_output_tokens == 128_000
    assert p.context_window == 400_000


def test_f54_unknown_xai_model_warns_once(caplog):
    """It silently took the cheapest tariff in the table, under-reporting session cost several times over with nothing in the log."""
    from pyutilz.llm.xai_provider import XAIProvider

    reset_log_throttles()
    p = XAIProvider.__new__(XAIProvider)
    with caplog.at_level("WARNING"):
        assert p._input_cost_per_1m("totally-unknown-model") == 0.20
    assert any("unknown" in r.getMessage() for r in caplog.records)


def test_f54_dated_grok_snapshot_resolves_by_prefix():
    from pyutilz.llm.xai_provider import XAIProvider

    p = XAIProvider.__new__(XAIProvider)
    assert p._input_cost_per_1m("grok-4-0709-2026-01-01") == 3.00


# ---------------------------------------------------------------------------
# F55 -- OpenRouter uptime normalisation
# ---------------------------------------------------------------------------


def test_f55_one_point_zero_in_a_percentage_payload_is_one_percent():
    """`v / 100 if v > 1 else v` resolved the single ambiguous value in the unsafe direction, so
    a nearly-dead backend reporting 1% uptime read as 100% and kept receiving traffic."""
    from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

    out = _summarize_endpoints([
        {"provider_name": "dying", "uptime_last_30m": 1.0},
        {"provider_name": "healthy", "uptime_last_30m": 99.9},
    ])
    dying = out["endpoints"][0]
    assert dying["uptime_30m"] == pytest.approx(0.01)


def test_f55_fraction_payload_keeps_one_point_zero_as_perfect():
    from pyutilz.llm.openrouter_provider._health import _summarize_endpoints

    out = _summarize_endpoints([
        {"provider_name": "a", "uptime_last_30m": 1.0},
        {"provider_name": "b", "uptime_last_30m": 0.98},
    ])
    assert out["endpoints"][0]["uptime_30m"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F58 -- token_counter encoding cache
# ---------------------------------------------------------------------------


def test_f58_encoding_cache_survives_concurrent_mutation():
    """popitem() at the cap could raise KeyError between the len() check and the pop, surfacing
    as a spurious failure inside a token-budget computation."""
    from pyutilz.llm import token_counter as tc

    if not tc._HAS_TIKTOKEN:
        pytest.skip("tiktoken not installed")
    errors = []

    def worker(n):
        try:
            for i in range(200):
                tc._encoding_for_model(f"model-{n}-{i}")
        except Exception as exc:  # the point of the test is that nothing escapes
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(tc._encoding_cache) <= tc._ENCODING_CACHE_MAX_SIZE


# ---------------------------------------------------------------------------
# F12 / F41 / F59 -- text.similarity
# ---------------------------------------------------------------------------

def test_f12_exact_suffix_match_scores_like_an_exact_prefix_match():
    """`range(L-S)` never tested the SUFFIX window, so MADRID vs REALMADRID scored 0.633."""
    assert sentences_similarity(["MADRID"], ["REALMADRID"]) == pytest.approx(0.80)
    assert sentences_similarity(["bcd"], ["abcd"]) == pytest.approx(0.875)


def test_f41_single_character_overlap_no_longer_scores_above_0_9():
    """The prefix rule fired before the min-length guard, so a one-letter token outscored a
    correct suffix match against any word starting with that letter."""
    assert sentences_similarity(["A"], ["ANDERSON"]) < 0.9


def test_f41_genuine_multi_character_prefix_still_scores_high():
    assert sentences_similarity(["REAL"], ["REALLYBIGCLUB"]) > 0.9


def test_f59_normalize_sentence_is_order_stable():
    """`list(set(...))` varied with PYTHONHASHSEED, and the greedy matcher's >= tie-break is
    order-sensitive, so the same two names could score differently in two processes."""
    assert normalize_sentence("NEW YORK NEW YORK") == ["NEW", "YORK"]
    assert normalize_sentence("ZULU ALPHA MIKE") == ["ZULU", "ALPHA", "MIKE"]


def test_f59_normalize_sentence_drops_empty_tokens():
    assert "" not in normalize_sentence("")


# ---------------------------------------------------------------------------
# F13 -- text entropy
# ---------------------------------------------------------------------------

def test_f13_raw_entropy_is_never_negative():
    """Normalizing by the number of DISTINCT prefixes made each "probability" exceed 1, flipping
    the log2 term's sign; Shannon entropy is never negative."""
    raw, _rate = compute_entropy_stats("abcabcabcabc", 0)
    assert raw >= 0.0


def test_f13_first_order_raw_entropy_matches_the_analytic_value():
    raw, _rate = compute_entropy_stats("abcabcabcabc", 1)
    assert raw == pytest.approx(1.5726236638951638)


# ---------------------------------------------------------------------------
# F14 / F62 -- secret scrubbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dsn",
    [
        "redis://:sup3rs3cret@localhost:6379/0",
        "postgresql://:sup3rs3cret@db.internal:5432/prod",
        "amqp://:sup3rs3cret@rabbit",
    ],
)
def test_f14_empty_username_dsn_is_redacted(dsn):
    """Empty-user is the standard Redis/valkey password-only URL form, and both functions
    returned it verbatim."""
    assert "sup3rs3cret" not in sanitize_dsn(dsn)
    assert "sup3rs3cret" not in redact_secrets(dsn)


def test_f14_basic_auth_header_is_redacted():
    """TELEGRAM_TOKEN_RE knew only Bearer, so the base64 user:pass was printed in full."""
    out = redact_secrets("Authorization: Basic dXNlcjpwYXNz")
    assert "dXNlcjpwYXNz" not in out


def test_f14_bearer_header_still_redacted():
    assert redact_secrets("Send Bearer ghp_secret to authenticate.") == "Send Bearer *** to authenticate."


def test_f62_two_dsns_in_one_message_both_survive_masked():
    """The greedy `.+` anchored to the LAST @ deleted the second DSN, the connective text and the
    first host, then attributed the surviving host to the first user."""
    out = sanitize_dsn("a postgres://u:p1@h1/d and postgres://u2:p2@h2/d")
    assert "h1" in out and "h2" in out and " and " in out
    assert "p1" not in out and "p2" not in out


def test_f62_at_containing_password_is_still_fully_masked():
    assert sanitize_dsn("postgresql://user:p@ss@host/db") == "postgresql://user:***@host/db"


# ---------------------------------------------------------------------------
# F37 -- json_pg_dumps
# ---------------------------------------------------------------------------

def test_f37_literal_backslash_u0000_text_survives():
    """Editing the SERIALIZED text could not tell the escape from a value that legitimately
    contains those six characters, so a Windows path or regex crashed the serializer."""
    value = "path" + _BS + "u0000literal"
    assert json_pg_dumps({"a": value}).adapted == {"a": value}


def test_f37_doubled_backslash_variant_survives():
    value = "p" + _BS * 2 + "u0000"
    assert json_pg_dumps({"a": value}).adapted == {"a": value}


def test_f37_real_nul_is_still_stripped():
    assert json_pg_dumps({"a": "x" + chr(0) + "y"}).adapted == {"a": "xy"}


def test_f37_nul_in_a_key_is_stripped_too():
    assert json_pg_dumps({"k" + chr(0): 1}).adapted == {"k": 1}


def test_f37_nonfinite_floats_still_become_null():
    assert json_pg_dumps({"a": float("inf")}).adapted == {"a": None}


# ---------------------------------------------------------------------------
# F38 / F39 -- webtext
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", None])
def test_f38_remove_videos_returns_its_input_for_falsy_text(value):
    """It fell off the `if text:` branch and returned None, which clean_description then
    dereferenced as a string."""
    assert remove_videos(value) == value


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_f38_clean_description_handles_blank_input(value):
    """Whitespace-only rows are routine for scraped text (fix_html strips them to ""), and one
    blank listing aborted the whole batch with a TypeError."""
    from pyutilz.text.strings.webtext import clean_description

    assert clean_description(value) == ""


def test_f39_named_and_numeric_entities_are_decoded():
    """xml.sax.saxutils.unescape knew only &amp;/&lt;/&gt;, leaving the most frequent entities in
    scraped HTML untouched."""
    out = unescape_html("Caf&eacute; &amp; bar &#39;x&#39; &nbsp;")
    assert "&eacute;" not in out
    assert "&#39;" not in out
    assert "&nbsp;" not in out
    assert "&" in out


def test_f39_quot_is_decoded():
    assert unescape_html("&quot;hi&quot;") == '"hi"'


# ---------------------------------------------------------------------------
# F40 / F63 -- tokenizers capitalization flags
# ---------------------------------------------------------------------------


def test_f40_min_morpheme_length_above_one_does_not_crash(monkeypatch):
    """FIRSTLETTER_CAPITAL/ALLLETTERS_CAPITAL were assigned only inside the `j == 1` branch, so
    starting the loop at j=2 raised NameError on the first word."""
    nltk = pytest.importorskip("nltk")
    _require_spacy_model()
    from pyutilz.text.tokenizers import AdvancedTokenizer

    tok = AdvancedTokenizer()
    monkeypatch.setattr(type(tok), "MIN_MORPHEME_LENGTH", 2, raising=False)
    monkeypatch.setattr(nltk, "sent_tokenize", lambda text: [text])
    monkeypatch.setattr(nltk, "word_tokenize", lambda text: text.split())
    tok.tokenize("Hello World")
    assert sum(tok.NUM_FIRSTLETTER_CAPITAL.values()) > 0


def test_f63_capitalization_flags_describe_the_word_not_the_offset(monkeypatch):
    """Both flags were recomputed at every start offset, so "aBCDE" -- neither capitalized nor
    all-caps -- contributed to both counters."""
    nltk = pytest.importorskip("nltk")
    _require_spacy_model()
    from pyutilz.text.tokenizers import AdvancedTokenizer

    tok = AdvancedTokenizer()
    monkeypatch.setattr(nltk, "sent_tokenize", lambda text: [text])
    monkeypatch.setattr(nltk, "word_tokenize", lambda text: text.split())
    tok.tokenize("aBCDE")
    assert sum(tok.NUM_FIRSTLETTER_CAPITAL.values()) == 0
    assert sum(tok.NUM_ALLLETTERS_CAPITAL.values()) == 0


# ---------------------------------------------------------------------------
# F60 -- find_between
# ---------------------------------------------------------------------------

def test_f60_explicit_zero_end_index_means_an_empty_window():
    """`if not idx2` conflated an explicit 0 with "unset" and searched the whole string."""
    assert find_between("abcXdefYghi", "X", "Y", 0, 0) is None


def test_f60_nonzero_window_still_works():
    assert find_between("abcXdefYghi", "X", "Y") == "def"
    assert find_between("abcXdefYghi", "X", "Y", 0, 5) is None


def test_f60_empty_start_honours_idx1():
    """`p1 = 0` discarded idx1 and returned content from before the requested start."""
    assert find_between("abcXdef", "", "X", 3) == ""
    assert find_between("abcXdef", "", "X") == "abc"


# ---------------------------------------------------------------------------
# F61 -- fix_dashes
# ---------------------------------------------------------------------------

def test_f61_multi_space_runs_far_from_a_dash_are_preserved():
    """An unconditional collapse destroyed aligned tables, indented code and any ASCII layout in
    a document that contained no dash at all."""
    assert fix_dashes("Column A    Column B") == "Column A    Column B"


def test_f61_spaces_around_a_replacement_dash_are_still_collapsed():
    assert fix_dashes("a  —  b") == "a - b"
    assert fix_dashes("a – b") == "a - b"
