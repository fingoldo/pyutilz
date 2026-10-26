from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


PRINTED_ADVICE_TESTS = {
    "src/pyutilz/core/serialization.py::_decompress_capped#1": "tests/test_security_audit_20260903.py::test_decompression_cap_error_names_a_working_unbounded_escape_hatch",
    "src/pyutilz/core/pythonlib/filesystem.py::_read_verified_pickle#1": "tests/test_printed_advice_messages.py::test_load_file_refusal_names_allow_unverified",
    "src/pyutilz/database/db/schema.py::_alternate_values_clause#1": "tests/test_printed_advice_messages.py::test_alternate_values_fragment_refusal_names_raw_sql",
    "src/pyutilz/database/db/upsert.py::build_upsert_query#1": "tests/test_printed_advice_messages.py::test_upsert_raw_fragment_refusal_names_raw_sql",
    "src/pyutilz/text/strings/configfiles.py::_normalise_encryption#1": "tests/test_printed_advice_messages.py::test_xor_encryption_warning_names_base64_spelling",
    "src/pyutilz/dev/freevar_analysis.py::split_out_module#1": "tests/test_dev_freevar_analysis.py::test_refusal_to_split_names_back_import_ok_escape_hatch",
    "src/pyutilz/dev/logginglib.py::init_logging#1": "tests/test_logginglib.py::TestInitLoggingStderrAdvice::test_failed_stderr_reconfigure_is_logged_with_its_errors_replace_note",
    "src/pyutilz/llm/anthropic_provider.py::AnthropicProvider.__init__#1": "tests/llm/test_printed_advice_coverage_anthropic_provider.py::test_anthropic_missing_key_names_pass_api_key",
    "src/pyutilz/llm/deepseek_provider.py::DeepSeekProvider.__init__#1": "tests/llm/test_deepseek_provider.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/gemini_provider.py::GeminiProvider.__init__#1": "tests/test_gemini_provider.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openai_provider.py::OpenAIProvider.__init__#1": "tests/llm/test_printed_advice_coverage_openai_provider.py::test_openai_missing_key_names_pass_api_key",
    "src/pyutilz/llm/openrouter_batch.py::OpenRouterBatchClient.__init__#1": "tests/test_llm_openrouter_batch.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openrouter_decisions.py::build_payload#1": "tests/test_llm_openrouter_decisions.py::test_build_payload_moving_alias_error_names_allow_moving_alias_opt_in",
    "src/pyutilz/llm/openrouter_decisions.py::_resolve_api_key#1": "tests/test_llm_openrouter_decisions.py::test_resolve_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/openrouter_provider/_provider.py::OpenRouterProvider.__init__#1": "tests/llm/test_openrouter_provider.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/llm/xai_provider.py::XAIProvider.__init__#1": "tests/llm/test_xai_provider.py::test_missing_api_key_error_names_pass_api_key_fallback",
    "src/pyutilz/system/config.py::TomlLiveConfig.get#1": "tests/test_system_audit_fixes.py::test_lossy_int_cast_warning_names_type_float_fix",
    "src/pyutilz/system/distributed.py::_content_version#1": "tests/test_distributed.py::test_content_version_without_a_module_file_names_version_kwarg",
}


def test_every_printed_advice_has_a_test():
    """The gate fails on an advising message with no entry, a stale entry, or an entry naming no test; each named test must exist."""
    # py-ci-shared requires python>=3.9 and the requirements marker skips it on the 3.8 legs, so skip there explicitly;
    # on 3.9+ a missing install still fails the import instead of silently dropping the gate.
    if sys.version_info < (3, 9):
        pytest.skip("py-ci-shared requires python>=3.9")
    from py_ci_shared.printed_advice import assert_printed_advice_registered

    source_files = sorted(p for p in (REPO_ROOT / "src" / "pyutilz").rglob("*.py") if "__pycache__" not in p.parts)
    assert_printed_advice_registered(source_files, REPO_ROOT, PRINTED_ADVICE_TESTS)
    import importlib.util

    for key, target in PRINTED_ADVICE_TESTS.items():
        path, _, test_name = target.partition("::")
        spec = importlib.util.spec_from_file_location(f"_printed_advice_target_{len(key)}", REPO_ROOT / path)
        assert spec is not None and spec.loader is not None, f"{key} names a test file that does not exist: {target}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        found = module
        for part in test_name.split("::"):
            assert hasattr(found, part), f"{key} names a test that does not exist: {target}"
            found = getattr(found, part)
