"""OpenRouter says whether a call REPORTED its cost, not only what the cost was.

`last_actual_cost_usd` is 0.0 both for a call OpenRouter reported as free and for a response that carried no
`usage.cost` at all, so a caller could not tell a real $0 from a missing figure (the realtime pipeline's re-audit
RA2-C11-8 had to guess from a ``:free`` model suffix). `last_actual_cost_reported` is True only when `usage.cost` was a
number, and is per-call state, reset with the rest of the call's metadata.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
pytest.importorskip("pydantic")

# Imported after the optional-dependency skips, which is what they are for.
from tests.test_llm_openrouter import _provider


def test_the_call_says_whether_a_cost_was_reported():
    p = _provider()
    p._reset_per_call_state()
    p._track_provider_specific_usage({"prompt_tokens": 10})
    assert (p.last_actual_cost_usd, p.last_actual_cost_reported) == (0.0, False), "a response with no cost field read as a reported $0"
    p._track_provider_specific_usage({"cost": 0})
    assert (p.last_actual_cost_usd, p.last_actual_cost_reported) == (0.0, True), "a reported $0 read as missing"


def test_the_flag_is_per_call_state():
    p = _provider()
    p._reset_per_call_state()
    p._track_provider_specific_usage({"cost": 0.002})
    assert p.last_actual_cost_reported is True
    p._reset_per_call_state()
    assert p.last_actual_cost_reported is False, "the previous call's report leaked into the next call"


@pytest.mark.parametrize("cost", [True, "0.01", None])
def test_only_a_number_counts_as_a_reported_cost(cost):
    p = _provider()
    p._reset_per_call_state()
    p._track_provider_specific_usage({"cost": cost})
    assert p.last_actual_cost_reported is False
