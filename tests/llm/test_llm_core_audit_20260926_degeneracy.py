from __future__ import annotations
import pytest


class TestRepetitionLoopMidUnit:
    @pytest.mark.parametrize("cut", ["", "H", "Hm", "Hmm", "Hmm."])
    def test_loop_found_when_the_stream_ends_mid_unit(self, cut):
        from pyutilz.llm.degeneracy import repetition_loop

        unit = repetition_loop("Hmm. " * 400 + cut)
        assert unit is not None
        assert sorted(unit.replace(" ", "")) == sorted("Hmm.")
