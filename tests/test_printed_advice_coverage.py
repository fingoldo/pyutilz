"""Every message that tells the reader to do something is followed by a test that does it and sees the effect.

The py-ci-shared gate ``printed_advice`` finds each advising literal in ``src/pyutilz`` and keys it
``<path>::<scope>#<n>``. ``PRINTED_ADVICE_TESTS`` maps each key to the test that performs the advice. A new
advising message, or a key whose message moved or was removed, fails the meta test below until the table is updated.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PRINTED_ADVICE_TESTS = {
    "src/pyutilz/core/serialization.py::_decompress_capped#1": "tests/test_security_audit_20260903.py::test_decompression_cap_error_names_a_working_unbounded_escape_hatch",
    "src/pyutilz/core/pythonlib/filesystem.py::_read_verified_pickle#1": "tests/test_printed_advice_messages.py::test_load_file_refusal_names_allow_unverified",
    "src/pyutilz/database/db/schema.py::_alternate_values_clause#1": "tests/test_printed_advice_messages.py::test_alternate_values_fragment_refusal_names_raw_sql",
    "src/pyutilz/database/db/upsert.py::build_upsert_query#1": "tests/test_printed_advice_messages.py::test_upsert_raw_fragment_refusal_names_raw_sql",
    "src/pyutilz/dev/code_audit/registry.py::register_scanner#1": "tests/code_audit/test_registry_invariants.py::test_duplicate_scanner_registration_error_names_allow_override",
    "src/pyutilz/text/strings/configfiles.py::_normalise_encryption#1": "tests/test_printed_advice_messages.py::test_xor_encryption_warning_names_base64_spelling",
    "src/pyutilz/dev/freevar_analysis.py::split_out_module#1": "tests/test_dev_freevar_analysis.py::test_refusal_to_split_names_back_import_ok_escape_hatch",
    "src/pyutilz/dev/logginglib.py::init_logging#1": "tests/test_logginglib.py::TestInitLoggingStderrAdvice::test_failed_stderr_reconfigure_is_logged_with_its_errors_replace_note",
    "src/pyutilz/llm/anthropic_provider.py::AnthropicProvider.__init__#1": "tests/test_printed_advice_coverage.py::test_anthropic_missing_key_names_pass_api_key",
    "src/pyutilz/llm/deepseek_provider.py::DeepSeekProvider.__init__#1": "tests/test_llm_deepseek.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/gemini_provider.py::GeminiProvider.__init__#1": "tests/test_gemini_provider.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openai_provider.py::OpenAIProvider.__init__#1": "tests/test_printed_advice_coverage.py::test_openai_missing_key_names_pass_api_key",
    "src/pyutilz/llm/openrouter_batch.py::OpenRouterBatchClient.__init__#1": "tests/test_llm_openrouter_batch.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openrouter_decisions.py::build_payload#1": "tests/test_llm_openrouter_decisions.py::test_build_payload_moving_alias_error_names_allow_moving_alias_opt_in",
    "src/pyutilz/llm/openrouter_decisions.py::_resolve_api_key#1": "tests/test_llm_openrouter_decisions.py::test_resolve_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openrouter_provider/_provider.py::OpenRouterProvider.__init__#1": "tests/test_llm_openrouter.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/xai_provider.py::XAIProvider.__init__#1": "tests/test_llm_xai.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/system/config.py::TomlLiveConfig.get#1": "tests/test_system_audit_fixes.py::test_lossy_int_cast_warning_names_type_float_fix",
    "src/pyutilz/system/distributed.py::_content_version#1": "tests/test_distributed.py::test_content_version_without_a_module_file_names_version_kwarg",
}


def test_anthropic_missing_key_names_pass_api_key():
    from pyutilz.llm.anthropic_provider import AnthropicProvider

    with patch("pyutilz.llm.anthropic_provider.get_llm_settings", return_value=SimpleNamespace(anthropic_api_key=None)):
        with pytest.raises(ValueError, match=r"pass api_key="):
            AnthropicProvider(api_key=None)


def test_openai_missing_key_names_pass_api_key():
    from pyutilz.llm.openai_provider import OpenAIProvider

    mock_settings = MagicMock()
    mock_settings.openai_api_key = None
    with patch("pyutilz.llm.openai_provider.get_llm_settings", return_value=mock_settings):
        with pytest.raises(ValueError, match=r"pass api_key="):
            OpenAIProvider(api_key=None)


def test_every_printed_advice_has_a_test():
    """The gate fails on an advising message with no entry, a stale entry, or an entry naming no test; each named test must exist."""
    # py-ci-shared requires python>=3.9 and the requirements marker skips it on the 3.8 legs, so skip there explicitly;
    # on 3.9+ a missing install still fails the import instead of silently dropping the gate.
    if sys.version_info < (3, 9):
        pytest.skip("py-ci-shared requires python>=3.9")
    from py_ci_shared.printed_advice import assert_printed_advice_registered

    source_files = sorted(p for p in (REPO_ROOT / "src" / "pyutilz").rglob("*.py") if "__pycache__" not in p.parts)
    assert_printed_advice_registered(source_files, REPO_ROOT, PRINTED_ADVICE_TESTS)
    for key, target in PRINTED_ADVICE_TESTS.items():
        path, _, test_name = target.partition("::")
        leaf = test_name.rsplit("::", 1)[-1]
        assert f"def {leaf}(" in (REPO_ROOT / path).read_text(encoding="utf-8"), f"{key} names a test that does not exist: {target}"  # audit-exempt: source_text_assertion -- checks that a named test exists in a file (a name lookup), not the behaviour of the code
