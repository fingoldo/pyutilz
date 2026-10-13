"""A Claude model the table lacks is priced from OpenRouter's listing of it, not from the unknown-model default."""

from __future__ import annotations

import pytest

from pyutilz.llm import _claude_models as cm
from pyutilz.llm.openrouter_provider import _catalogue

_HAIKU_LISTING = {
    "anthropic/claude-haiku-5.5": {
        "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
        "top_provider": {"max_completion_tokens": 128000},
        "context_length": 1000000,
    },
    "anthropic/claude-haiku-5.5:batch": {
        "pricing": {"prompt": "0.00000005", "completion": "0.00000025"},
        "top_provider": {"max_completion_tokens": 128000},
        "context_length": 1000000,
    },
    "~anthropic/claude-haiku-latest": {
        "pricing": {"prompt": "0.0000009", "completion": "0.000009"},
        "top_provider": {"max_completion_tokens": 1},
        "context_length": 1,
    },
}


@pytest.fixture(autouse=True)
def _fresh_lookup_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cm, "_openrouter_specs", {})
    monkeypatch.setattr(cm, "_warned_unknown", set())


def _listing(monkeypatch: pytest.MonkeyPatch, catalogue: dict) -> None:
    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", lambda *a, **k: catalogue)


def test_an_unlisted_claude_model_takes_its_openrouter_price_and_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    _listing(monkeypatch, _HAIKU_LISTING)
    spec = cm.claude_model_spec("claude-haiku-5-5", provider_label="Claude Code")
    assert spec.input_per_1m == pytest.approx(0.10)
    assert spec.output_per_1m == pytest.approx(0.50)
    assert spec.max_output == 128000
    assert spec.context_window == 1000000


def test_variants_and_aliases_are_not_the_models_own_price(monkeypatch: pytest.MonkeyPatch) -> None:
    _listing(monkeypatch, {k: v for k, v in _HAIKU_LISTING.items() if k != "anthropic/claude-haiku-5.5"})
    spec = cm.claude_model_spec("claude-haiku-5-5")
    assert spec == cm.UNKNOWN_CLAUDE_MODEL, "only :batch and ~latest listings exist, neither is the model's own"


def test_a_failed_catalogue_falls_back_to_the_default_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _listing(monkeypatch, {})
    assert cm.claude_model_spec("claude-haiku-5-5") == cm.UNKNOWN_CLAUDE_MODEL


def test_a_model_the_table_lists_is_never_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):  # the table row must answer without the network
        raise AssertionError("catalogue consulted for a table row")

    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", boom)
    assert cm.claude_model_spec("claude-sonnet-5").input_per_1m == cm.resolve_claude_model("claude-sonnet-5")[1].input_per_1m


def test_the_lookup_happens_once_per_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _HAIKU_LISTING

    monkeypatch.setattr(_catalogue, "_fetch_models_catalogue", counting)
    cm.claude_model_spec("claude-haiku-5-5")
    cm.claude_model_spec("claude-haiku-5-5")
    assert calls["n"] == 1


def test_the_listed_price_has_no_default_so_an_unlisted_model_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _listing(monkeypatch, _HAIKU_LISTING)
    assert cm.claude_model_listed_price("claude-haiku-5-5") == pytest.approx((0.10, 0.50))
    assert cm.claude_model_listed_price("claude-sonnet-5") is not None, "a table row prices without the network"
    _listing(monkeypatch, {})
    assert cm.claude_model_listed_price("claude-no-such-model-9") is None
