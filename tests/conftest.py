import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# ─── Live LLM-provider test infrastructure ──────────────────────────
# Live tests are gated by ``@pytest.mark.live`` AND a per-provider
# API key. The ``live`` marker is OPT-IN: default ``pytest`` skips
# every live test even when the key is present, because they cost
# real money. To execute them, pass ``--run-live``:
#
#   pytest --run-live tests/test_llm_live.py
#   pytest --run-live -m live
#
# When ``--run-live`` is set, individual tests still skip cleanly
# if their per-provider key is missing (so a contributor without
# every account can still run the subset they have keys for).


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run @pytest.mark.live tests (real LLM API calls -- costs money)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(
        reason="live LLM-provider test -- pass --run-live to execute"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def _require_provider_key(env_var: str, provider_name: str) -> str:
    """Resolve an API key from env or skip the test cleanly."""
    val = os.environ.get(env_var)
    if not val:
        # Try LLMSettings (.env file). Skip on any settings error.
        try:
            from pyutilz.llm.config import get_llm_settings
            settings = get_llm_settings()
            field = env_var.lower()
            secret = getattr(settings, field, None)
            if secret is not None and secret.get_secret_value():
                return secret.get_secret_value()
        except Exception:
            pass
        pytest.skip(f"{env_var} not set -- skipping {provider_name} live test")
    return val


@pytest.fixture
def openrouter_key() -> str:
    """API key for live OpenRouter tests; skips if absent."""
    return _require_provider_key("OPENROUTER_API_KEY", "OpenRouter")


@pytest.fixture
def anthropic_key() -> str:
    """API key for live Anthropic tests; skips if absent."""
    return _require_provider_key("ANTHROPIC_API_KEY", "Anthropic")


@pytest.fixture
def openai_key() -> str:
    """API key for live OpenAI tests; skips if absent."""
    return _require_provider_key("OPENAI_API_KEY", "OpenAI")


@pytest.fixture
def deepseek_key() -> str:
    """API key for live DeepSeek tests; skips if absent."""
    return _require_provider_key("DEEPSEEK_API_KEY", "DeepSeek")


@pytest.fixture
def xai_key() -> str:
    """API key for live xAI Grok tests; skips if absent."""
    return _require_provider_key("XAI_API_KEY", "xAI")


@pytest.fixture
def gemini_key() -> str:
    """API key for live Gemini tests; skips if absent."""
    return _require_provider_key("GEMINI_API_KEY", "Gemini")


# Default per-test spend ceiling for live calls. A live test going
# over this is almost always a bug (e.g. accidental 100k-token prompt).
LIVE_TEST_MAX_COST_USD = 0.005


@pytest.fixture
def assert_under_budget():
    """Assert a callable's resulting last-call cost stays under the cap.

    Usage:
        async def test_real(openrouter_key, assert_under_budget):
            p = OpenRouterProvider(api_key=openrouter_key, model="...")
            await p.generate("hi", max_tokens=5)
            assert_under_budget(p.last_actual_cost_usd)

    Override the cap per-test by passing ``cap=`` to the call.
    """
    def _check(actual_cost: float, cap: float = LIVE_TEST_MAX_COST_USD) -> None:
        assert actual_cost <= cap, (
            f"Live test cost ${actual_cost:.6f} exceeds budget cap ${cap:.6f}. "
            "Likely a bug (huge prompt, wrong model, infinite-output) -- "
            "treat as a regression rather than raising the cap."
        )
    return _check


@pytest.fixture
def sample_df():
    """Simple test DataFrame"""
    return pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def mixed_types_df():
    """DataFrame with various dtypes for optimization testing"""
    return pd.DataFrame({
        'large_int': np.array([1, 2, 3], dtype=np.int64),
        'small_int': [1, 2, 3],
        'float64_col': np.array([1.0, 2.0, 3.0], dtype=np.float64),
        'category': pd.Categorical(['a', 'b', 'a']),
        'object_col': ['x', 'y', 'z']
    })


@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def float_with_integers_df():
    """DataFrame with float columns that have no fractional part"""
    return pd.DataFrame({
        'float_int': [1.0, 2.0, 3.0, 4.0],
        'float_real': [1.1, 2.2, 3.3, 4.4],
        'float_with_nan': [1.0, 2.0, np.nan, 4.0]
    })


@pytest.fixture
def constant_columns_df():
    """DataFrame with some constant columns"""
    return pd.DataFrame({
        'const1': [1, 1, 1, 1],
        'const2': ['a', 'a', 'a', 'a'],
        'varying': [1, 2, 3, 4],
        'varying2': ['a', 'b', 'c', 'd']
    })
