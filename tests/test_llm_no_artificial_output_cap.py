"""No LLM call may carry a numeric output cap we chose. The ratchet itself lives in py_ci_shared.llm_output_caps.

KNOWN_CAPS is the shrink-only list of caps that exist today, each with its reason. It is empty: every provider sends the
model's catalogue ceiling or omits max_tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("py_ci_shared.llm_output_caps", reason="py-ci-shared is not a pyutilz dependency; the gate runs where it is installed")

from py_ci_shared.llm_output_caps import (
    KnownCap,
    assert_each_known_cap_states_its_reason,
    assert_known_caps_only_shrink,
    assert_no_new_output_caps,
    numeric_output_caps,
)

REPO = Path(__file__).resolve().parents[1]
_LLM_ROOT = REPO / "src" / "pyutilz" / "llm"

KNOWN_CAPS: KnownCap = {}

_NAME_HINTS = ("_default_max_tokens", "max_output_tokens", "max_completion_tokens")
_DEFAULT_HINTS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _numeric_caps() -> set[tuple[str, str, int]]:
    return numeric_output_caps([_LLM_ROOT], repo_root=REPO, name_hints=_NAME_HINTS, default_hints=_DEFAULT_HINTS)


def test_no_new_numeric_output_cap_in_the_llm_package() -> None:
    assert_no_new_output_caps(_numeric_caps(), KNOWN_CAPS)


def test_known_caps_only_shrink() -> None:
    assert_known_caps_only_shrink(_numeric_caps(), KNOWN_CAPS)


def test_each_known_cap_states_its_reason() -> None:
    assert_each_known_cap_states_its_reason(KNOWN_CAPS)


# Behaviour: a provider whose model has no catalogue ceiling sends NO max_tokens; with a ceiling it sends the ceiling.
# A caller's own number is still honoured (the caller chose it; the gate is about numbers the library chooses).


def _openrouter(**kwargs):
    from pyutilz.llm.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(api_key="sk-test-not-real", **kwargs)  # pragma: allowlist secret


def test_a_provider_without_a_catalogue_entry_omits_max_tokens() -> None:
    p = _openrouter()
    p.model_name = "unlisted/model-x"
    body = p._request_body("hi", None, None, None, None, None, None)
    assert "max_tokens" not in body, body


def test_a_zero_request_with_no_ceiling_omits_max_tokens() -> None:
    p = _openrouter()
    p.model_name = "unlisted/model-x"
    body = p._request_body("hi", None, None, 0, None, None, None)
    assert "max_tokens" not in body, body


def test_a_provider_with_a_catalogue_ceiling_sends_that_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyutilz.llm.openrouter_provider import _catalogue as _catalogue_module

    catalogue = {"acme/not-in-static-table": {"top_provider": {"max_completion_tokens": 12345}}}
    monkeypatch.setattr(_catalogue_module, "_fetch_models_catalogue", lambda: catalogue)
    p = _openrouter()
    p.model_name = "acme/not-in-static-table"
    assert p.max_output_tokens == 12345
    body = p._request_body("hi", None, None, p.max_output_tokens, None, None, None)
    assert body.get("max_tokens") == 12345, body
