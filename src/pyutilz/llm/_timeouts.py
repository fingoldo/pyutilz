"""How long to wait for a model, derived from what was asked of it rather than from its name.

Carved out of ``openai_compat`` for the 1000-line budget; the policy is one coherent unit and every
number in it was set by a measurement that a guess had already got wrong. Mixed into
``OpenAICompatibleProvider``, which supplies ``_get_timeout`` and ``model_name``.
"""

from __future__ import annotations

from typing import Any


class DerivedTimeoutMixin:
    """The request-timeout policy: a model-name floor, raised by how much output the body asks for."""

    model_name: str

    def _get_timeout(self, model: str) -> float:  # pragma: no cover - supplied by the provider
        """The model-name timeout floor; the provider this mixes into supplies it."""
        raise NotImplementedError

    #: Seconds of headroom to allow per requested output token. A 30 tok/s floor is deliberately
    #: pessimistic: the cost of guessing low is every long generation dying at the same wall.
    _seconds_per_output_token: float = 1.0 / 30.0

    #: Hard ceiling on the DERIVED half of the timeout. `max_tokens` is a ceiling, not a target, and
    #: deriving a timeout from it as though the model will use all of it was measured 2026-09-08 to cost
    #: 71 minutes per stalled attempt on a 128,000-token cap, with ten retries behind it.
    #:
    #: The value must clear the largest answer that actually ARRIVES, or it cuts real work: set to 20
    #: minutes on first writing, it killed every `qwen/qwen3.8-flash` capture, because that model emits
    #: 70,783 tokens and needs 39 of them. The largest clean emission measured across a 19-model fleet is
    #: 85,694 (`z-ai/glm-5.3-flash`, `finish_reason="stop"`), which needs 2,856 s at the pessimistic
    #: 30 tok/s floor - so 3,000 s, sized from that measurement rather than from what feels patient.
    #:
    #: The trade is stated rather than hidden: a route that has gone silent now holds an attempt for up to
    #: 50 minutes. That is the wrong side to economise on, because a truncated answer is indistinguishable
    #: from a model that cannot do the task, while a slow one merely costs wall-clock. The model-name floor
    #: is never clamped, so a slow-tier model keeps whatever `_get_timeout` grants it.
    _max_derived_timeout_s: float = 3000.0

    def _timeout_for(self, body: dict[str, Any]) -> float:
        """Request timeout in seconds, taking the LARGER of the model heuristic and what this body asks for.

        `_get_timeout` classifies by model NAME, which cannot see how much output was requested: a
        `z-ai/glm-5.3-flash` asked for 54,853 tokens matches no "slow tier" substring and got the 240 s
        default, so every one of nine captures died in a ReadTimeout storm while the model was still
        generating - 35,185 tokens on the one that finished, which no 240 s budget could ever cover. The
        floor stays: a small request on a slow-tier model keeps its long allowance.

        A model name cannot see TIME TO FIRST TOKEN either, which is the same defect in another
        direction: z-ai/glm-5.3-flash thinks for 393 s before emitting any content (measured
        2026-09-13), so a streamed call left on the 240 s constructor timeout died mid-thought with the
        model working normally. Both transports take their timeout from here for that reason.

        The DERIVED half is capped by `_max_derived_timeout_s`, because a ceiling is not a target: pinning
        an arena to one upstream route asks for that route's full cap, and deriving 71 minutes per attempt
        from a 128,000-token cap turned one quiet route into an hours-long stall. Only the derived half is
        clamped - `_get_timeout`'s own answer still wins outright when it is larger.
        """
        base = self._get_timeout(self.model_name)
        requested = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        try:
            needed = float(requested) * self._seconds_per_output_token
        except (TypeError, ValueError):
            return base
        return max(base, min(needed, self._max_derived_timeout_s))
