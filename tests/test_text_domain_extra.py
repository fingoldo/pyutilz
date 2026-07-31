"""Regression tests for pyutilz.text.* (2026-07-21 audit fixes)."""

import logging

import pytest

from pyutilz.text.strings.textentropy import naive_entropy_rate
from pyutilz.text.strings._logproxy import _FacadeLoggerProxy
from pyutilz.text.humanizer import humanize
from pyutilz.text.similarity import sentences_similarity, sentences_similarity_numba


class TestJsonPgDumpsNonFiniteNormalization:
    def test_nan_and_infinity_normalized_regardless_of_backend(self):
        from pyutilz.text.strings.jsonutils import _normalize_nonfinite_floats
        import math

        obj = {"score": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "ok": 1.5}
        result = _normalize_nonfinite_floats(obj)
        assert result == {"score": None, "inf": None, "ninf": None, "ok": 1.5}

    def test_nested_structures_normalized(self):
        from pyutilz.text.strings.jsonutils import _normalize_nonfinite_floats

        obj = {"a": [1.0, float("nan"), {"b": float("inf")}]}
        result = _normalize_nonfinite_floats(obj)
        assert result == {"a": [1.0, None, {"b": None}]}


class TestReadConfigFileSectionCollision:
    def test_all_sections_prefixed_to_avoid_collision(self, tmp_path):
        from pyutilz.text.strings.configfiles import read_config_file

        ini = tmp_path / "cfg.ini"
        ini.write_text("[SectionA]\ntimeout = 30\n\n[SectionB]\ntimeout = 60\n", encoding="utf-8")
        obj = {}
        read_config_file(str(ini), obj, section=None)
        assert obj.get("sectiona_timeout") == 30
        assert obj.get("sectionb_timeout") == 60
        assert "timeout" not in obj

    def test_explicit_str_section_unaffected(self, tmp_path):
        from pyutilz.text.strings.configfiles import read_config_file

        ini = tmp_path / "cfg.ini"
        ini.write_text("[SectionA]\ntimeout = 30\n", encoding="utf-8")
        obj = {}
        read_config_file(str(ini), obj, section="SectionA")
        assert obj.get("timeout") == 30

    def test_invalid_section_type_raises(self, tmp_path):
        from pyutilz.text.strings.configfiles import read_config_file

        ini = tmp_path / "cfg.ini"
        ini.write_text("[SectionA]\nx = 1\n", encoding="utf-8")
        assert read_config_file(str(ini), {}, section=123) is None  # caught by the outer except, returns None


class TestSimilaritySurrogateHandling:
    def test_numba_handles_lone_surrogate_like_pure_python(self):
        pure = sentences_similarity(["\ud800"], ["\ud800"])
        numba = sentences_similarity_numba(["\ud800"], ["\ud800"])
        assert pure == 1.0
        assert numba == pytest.approx(pure)

    def test_normal_unicode_unaffected(self):
        pure = sentences_similarity(["привет мир"], ["привет мир"])
        numba = sentences_similarity_numba(["привет мир"], ["привет мир"])
        assert numba == pytest.approx(pure)


class TestLogProxyDunders:
    def test_repr_and_str_forward_to_real_logger(self):
        proxy = _FacadeLoggerProxy()
        real = proxy._resolve()
        assert repr(proxy) == repr(real)
        assert str(proxy) == str(real)


class TestStringsFacadeLiveGlobals:
    def test_inflect_engine_reflects_webtext_live_state(self):
        pytest.importorskip("inflect")
        import pyutilz.text.strings as strings_pkg
        import pyutilz.text.strings.webtext as webtext_mod

        webtext_mod.suffixize("job", 2)
        assert strings_pkg.inflect_engine is webtext_mod.inflect_engine
        assert strings_pkg.inflect_engine is not None

    def test_unknown_attribute_still_raises(self):
        import pyutilz.text.strings as strings_pkg

        with pytest.raises(AttributeError):
            strings_pkg.definitely_not_a_real_attribute


class TestHumanizeProtectedSpansSurviveDrift:
    def test_protected_content_preserved_despite_earlier_stage_shifting_offsets(self):
        import random

        text = "It's worth noting that PASSPHRASE-42 must appear verbatim."
        start = text.index("PASSPHRASE-42")
        end = start + len("PASSPHRASE-42")
        rng = random.Random(0)
        result = humanize(text, typo_count=5, rng=rng, protected_spans=[(start, end)])
        assert "PASSPHRASE-42" in result


class TestNaiveEntropyRateEmptyString:
    def test_empty_string_returns_zero_not_negative_zero(self):
        result = naive_entropy_rate("")
        assert result == 0.0
        import math
        assert math.copysign(1.0, result) == 1.0  # not -0.0
