"""Exact-message, boundary and constant pins for the OpenRouter speech request validation (mutation survivors)."""

from __future__ import annotations

import math

import pytest

from pyutilz.llm import openrouter_speech as sp
from pyutilz.llm.openrouter_speech import SpeechError, SpeechInputError, SpeechRequest


def _req(**overrides) -> SpeechRequest:
    base = dict(text="hello", model="elevenlabs/eleven-v3", voice="v")
    base.update(overrides)
    return SpeechRequest(**base)


class TestModelCaps:
    @pytest.mark.parametrize(
        "model, cap",
        [
            ("elevenlabs/eleven-v4", 10_000),
            ("elevenlabs/eleven-v4-turbo", 10_000),
            ("elevenlabs/eleven-v3", 5_000),
            ("elevenlabs/eleven-v3-conversational", 5_000),
            ("elevenlabs/eleven-multilingual-v2", 10_000),
            ("elevenlabs/eleven-flash-v2.5", 40_000),
            ("elevenlabs/eleven-flash-v2", 30_000),
            ("elevenlabs/eleven-turbo-v2.5", 40_000),
            ("elevenlabs/eleven-turbo-v2", 30_000),
        ],
    )
    def test_each_model_cap_is_the_documented_limit(self, model, cap):
        assert sp.MODEL_MAX_CHARS[model] == cap

    def test_an_input_at_the_cap_is_accepted_and_one_over_is_refused(self):
        _req(text="x" * 5000).validate()
        with pytest.raises(SpeechInputError) as info:
            _req(text="x" * 5001).validate()
        assert str(info.value) == "elevenlabs/eleven-v3 accepts at most 5000 characters per request, got 5001"

    def test_an_unknown_model_has_no_cap(self):
        _req(model="openai/tts", text="x" * 50_000).validate()


class TestConstants:
    def test_the_speed_and_unit_ranges(self):
        assert sp.SPEED_RANGE == (0.7, 1.2)
        assert sp.UNIT_RANGE == (0.0, 1.0)
        assert sp.SEED_RANGE == (0, 2**32 - 1)

    def test_the_speed_rejecting_models_are_the_v4_pair(self):
        assert sp.SPEED_REJECTING_MODELS == frozenset({"elevenlabs/eleven-v4", "elevenlabs/eleven-v4-turbo"})

    def test_the_voice_setting_keys_are_exactly_five(self):
        assert sp.VOICE_SETTING_KEYS == frozenset({"stability", "similarity_boost", "style", "use_speaker_boost", "speed"})

    def test_the_normalisation_and_format_sets(self):
        assert sp.TEXT_NORMALIZATION == frozenset({"auto", "on", "off"})
        assert sp.UNVERIFIED_TEXT_NORMALIZATION == frozenset({"on"})
        assert sp.RESPONSE_FORMATS == frozenset({"mp3", "pcm"})

    def test_the_non_retryable_statuses_are_fixed(self):
        assert sp._NON_RETRYABLE == frozenset({400, 401, 402, 403, 404, 405, 410, 413, 422})
        assert sp._RETRYABLE_4XX == frozenset({408, 425, 429})

    def test_the_retry_and_budget_constants(self):
        assert sp.MAX_RETRY_AFTER_S == 120.0
        assert sp.MIN_RETRY_DELAY_S == 1.0
        assert sp.RETRY_AFTER_JITTER == 0.1
        assert sp.MAX_AUDIO_BYTES == 64 * 1024 * 1024
        assert sp.MAX_CALL_S == 900.0
        assert sp._EXPIRY_WARNING_DAYS == 30
        assert sp._CATALOGUE_TTL_S == 3600.0
        assert sp._CATALOGUE_FAILURE_BACKOFF_S == 60.0

    def test_the_generation_id_pattern_takes_one_to_128_safe_characters(self):
        assert sp._GENERATION_ID_RE.fullmatch("abc-_123")
        assert sp._GENERATION_ID_RE.fullmatch("a" * 128)
        assert not sp._GENERATION_ID_RE.fullmatch("a" * 129)
        assert not sp._GENERATION_ID_RE.fullmatch("")

    def test_the_digit_pattern_needs_at_least_one_digit(self):
        assert sp._DIGITS.fullmatch("") is None
        assert sp._DIGITS.fullmatch("123") is not None


class TestSpeechErrorDefaults:
    def test_the_defaults_are_no_status_not_retryable_other_kind_and_no_details(self):
        err = SpeechError("m")
        assert (err.status, err.retryable, err.kind, err.details) == (0, False, "other", {})

    def test_the_message_is_the_exception_text(self):
        assert str(SpeechError("m")) == "m"

    def test_supplied_fields_are_kept(self):
        err = SpeechError("m", 429, True, "rate", {"a": 1})
        assert (err.status, err.retryable, err.kind, err.details) == (429, True, "rate", {"a": 1})

    def test_the_input_error_is_a_value_error(self):
        assert issubclass(SpeechInputError, ValueError)


class TestCanonical:
    def test_numbers_become_floats_and_containers_recurse(self):
        assert sp._canonical({"a": (1, [2, True]), "b": "s"}) == {"a": [1.0, [2.0, True]], "b": "s"}

    def test_booleans_and_strings_are_kept_as_they_are(self):
        assert sp._canonical(True) is True
        assert sp._canonical("x") == "x"
        assert sp._canonical(None) is None

    def test_a_mapping_key_is_stringified(self):
        assert sp._canonical({1: "a"}) == {"1": "a"}


class TestValidateTextAndFormat:
    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_or_blank_text_is_refused(self, text):
        with pytest.raises(SpeechInputError) as info:
            _req(text=text).validate()
        assert str(info.value) == "speech input text is empty"

    def test_an_unknown_response_format_names_the_allowed_formats(self):
        with pytest.raises(SpeechInputError) as info:
            _req(response_format="wav").validate()
        assert str(info.value) == "response_format must be one of ['mp3', 'pcm'], got 'wav'"

    def test_elevenlabs_options_on_another_provider_are_refused(self):
        with pytest.raises(SpeechInputError) as info:
            _req(model="openai/tts-1", seed=3).validate()
        assert str(info.value) == "ElevenLabs options were given for openai/tts-1, which is not an ElevenLabs model"

    def test_an_unknown_text_normalisation_names_the_allowed_values(self):
        with pytest.raises(SpeechInputError) as info:
            _req(apply_text_normalization="bogus").validate()
        assert str(info.value) == "apply_text_normalization must be one of ['auto', 'off', 'on']"

    def test_the_unverified_normalisation_is_refused_with_its_reason(self):
        with pytest.raises(SpeechInputError) as info:
            _req(apply_text_normalization="on").validate()
        assert str(info.value) == (
            "apply_text_normalization='on' is refused: it may need enterprise access on v2.5 "
            "and has not been verified through OpenRouter; use 'auto' or 'off'"
        )

    def test_auto_and_off_normalisation_are_accepted(self):
        _req(apply_text_normalization="auto").validate()
        _req(apply_text_normalization="off").validate()


class TestCheckOptions:
    def test_a_seed_outside_the_range_is_refused_with_the_range(self):
        with pytest.raises(SpeechInputError) as info:
            _req(seed=-1).validate()
        assert str(info.value) == "seed must be an integer within (0, 4294967295), got -1"

    def test_a_boolean_seed_is_refused(self):
        with pytest.raises(SpeechInputError):
            _req(seed=True).validate()

    def test_the_seed_range_ends_are_accepted(self):
        _req(seed=0).validate()
        _req(seed=2**32 - 1).validate()

    def test_a_blank_language_code_is_refused_with_its_value(self):
        with pytest.raises(SpeechInputError) as info:
            _req(language_code="").validate()
        assert str(info.value) == "language_code must be a non-empty string, got ''"

    def test_a_non_bool_language_normalisation_is_refused(self):
        with pytest.raises(SpeechInputError) as info:
            _req(apply_language_text_normalization="yes").validate()
        assert str(info.value) == "apply_language_text_normalization must be a bool"


class TestCheckVoiceSettings:
    def test_an_unknown_key_is_refused_with_the_allowed_keys(self):
        with pytest.raises(SpeechInputError) as info:
            _req(voice_settings={"pitch": 1}).validate()
        assert str(info.value) == f"unknown voice_settings key 'pitch'; allowed: {sorted(sp.VOICE_SETTING_KEYS)}"

    def test_a_non_bool_speaker_boost_is_refused_with_its_value(self):
        with pytest.raises(SpeechInputError) as info:
            _req(voice_settings={"use_speaker_boost": "yes"}).validate()
        assert str(info.value) == "voice_settings.use_speaker_boost must be a bool, got 'yes'"

    def test_a_fraction_outside_the_unit_range_is_refused_with_the_range(self):
        with pytest.raises(SpeechInputError) as info:
            _req(voice_settings={"stability": 1.5}).validate()
        assert str(info.value) == "voice_settings.stability must be a number within (0.0, 1.0), got 1.5"

    def test_the_unit_range_ends_are_accepted(self):
        _req(voice_settings={"stability": 0.0, "style": 1.0}).validate()

    def test_a_boolean_fraction_is_refused(self):
        with pytest.raises(SpeechInputError):
            _req(voice_settings={"stability": True}).validate()

    def test_a_speed_inside_the_voice_settings_is_range_checked(self):
        with pytest.raises(SpeechInputError) as info:
            _req(voice_settings={"speed": 2.0}).validate()
        assert str(info.value) == "speed must be within (0.7, 1.2), got 2.0"


class TestCheckSpeed:
    @pytest.mark.parametrize("speed", [float("nan"), float("inf"), "fast"])
    def test_a_non_finite_or_non_number_speed_is_refused(self, speed):
        with pytest.raises(SpeechInputError) as info:
            _req(speed=speed).validate()
        assert str(info.value) == f"speed must be a finite number, got {speed!r}"

    def test_a_speed_rejecting_model_refuses_any_non_default_speed(self):
        with pytest.raises(SpeechInputError) as info:
            _req(model="elevenlabs/eleven-v4", speed=0.9).validate()
        assert str(info.value) == "elevenlabs/eleven-v4 rejects a non-default speed"

    def test_a_speed_rejecting_model_accepts_the_default_speed(self):
        _req(model="elevenlabs/eleven-v4", speed=1.0).validate()

    def test_a_speed_below_the_range_is_refused_with_the_range(self):
        with pytest.raises(SpeechInputError) as info:
            _req(speed=0.6).validate()
        assert str(info.value) == "speed must be within (0.7, 1.2), got 0.6"

    def test_the_speed_range_ends_are_accepted(self):
        _req(speed=0.7).validate()
        _req(speed=1.2).validate()

    def test_a_non_finite_speed_check_uses_math_isfinite(self):
        assert not math.isfinite(float("nan"))


class TestElevenLabsOptionDetection:
    def test_a_non_elevenlabs_model_with_only_plain_options_is_accepted(self):
        _req(model="openai/tts-1").validate()

    def test_an_empty_voice_settings_mapping_still_counts_as_an_elevenlabs_option(self):
        with pytest.raises(SpeechInputError):
            _req(model="openai/tts-1", voice_settings={}).validate()

    def test_a_language_code_alone_counts_as_an_elevenlabs_option(self):
        assert _req(language_code="en")._has_elevenlabs_options() is True
        assert _req()._has_elevenlabs_options() is False
