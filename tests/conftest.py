import os
import warnings
import pytest
import pandas as pd
import numpy as np

# ===========================================================================
# Warning policy (F12, 2026-09-02)
# ===========================================================================
# pyproject.toml's [tool.pytest.ini_options] carries no ``filterwarnings`` key, so pyutilz's own
# DeprecationWarnings -- the ones test_db_extra.py's PascalCase->snake_case tests and
# test_meta/test_deprecation_warnings_present.py exist to police -- were indistinguishable from
# third-party noise (weasel/Click, GPUtil, ...). Registered here rather than in pyproject.toml so
# the policy sits next to the suite it governs.
#
# Ratchet form: everything pyutilz itself emits is an ERROR (a stray internal DeprecationWarning
# fails the test that triggers it); third-party DeprecationWarning/PendingDeprecationWarning is
# ignored, since it is neither actionable here nor suppressible at the source. Order matters --
# pytest applies ini filters LAST-to-first, so the pyutilz-specific ``error`` must come after the
# broad ``ignore`` to win.
def pytest_configure(config):
    """Install the suite's warning policy (see the block comment above)."""
    for spec in (
        "ignore::DeprecationWarning",
        "ignore::PendingDeprecationWarning",
        "error::DeprecationWarning:pyutilz",
        "error::DeprecationWarning:pyutilz.*",
    ):
        config.addinivalue_line("filterwarnings", spec)


# ===========================================================================
# thinc / pytest-randomly seed-overflow compat shim (ported from mlframe)
# ===========================================================================
# pytest-randomly IS installed and active in the reference environment (plugin banner shows
# ``randomly-4.0.1``); the 2026-09-02 audit's claim that it was absent came from a run that
# passed ``-p no:randomly``. It is declared in requirements-dev.txt (not pyproject's [dev]
# extra, which is published metadata) so the order-randomisation the comments in test_image.py
# and test_pandaslib_extra.py rely on is a documented part of the contributor environment
# rather than an accident of whatever happens to be installed. The shim below is therefore live
# and load-bearing, not dead infrastructure.
# Root cause: thinc (spacy/explosion.ai dep) ships a
# ``pytest_randomly.random_seeder`` entry point -> ``thinc.util.fix_random_seed``,
# which calls ``numpy.random.seed(seed)`` WITHOUT the ``% 2**32`` clamp that
# pytest-randomly applies in its OWN ``_reseed``. pytest-randomly invokes every
# registered seeder with ``seed = randomly_seed + crc32(nodeid)`` -- the sum
# easily exceeds 2**32 -> ``ValueError: Seed must be between 0 and 2**32 - 1``
# at fixture setup, cascading to every test. Must patch at conftest IMPORT time
# (module level), not in a fixture: pytest-randomly resolves its entry points
# via ``e.load()`` before any fixture runs, so by then the wrapper must already
# be the bound ``thinc.util.fix_random_seed``. Drop once thinc clamps upstream.
try:
    import thinc.util as _thinc_util  # import placed after module-level setup on purpose (E402 is not in this repo's rule set, so the directive that used to sit here suppressed nothing)
    _thinc_original_fix = _thinc_util.fix_random_seed

    def _thinc_clamped_fix_random_seed(seed: int = 0) -> None:
        try:
            return _thinc_original_fix(int(seed) % (2**32))
        except Exception as _seed_err:
            # thinc transitively calls cupy.random.seed, which can crash on
            # boxes where cupy is installed but curand init fails. Fall back to
            # seeding python+numpy directly so determinism still holds.
            import random
            random.seed(int(seed) % (2**32))
            np.random.seed(int(seed) % (2**32))
            warnings.warn(
                f"thinc.util.fix_random_seed raised {_seed_err!r}; fell back to " f"seeding python+numpy directly (cupy random state unset).",
                RuntimeWarning,
                stacklevel=2,
            )

    _thinc_util.fix_random_seed = _thinc_clamped_fix_random_seed
    try:  # if pytest-randomly already cached the entry points, swap ours in
        import pytest_randomly as _pr  # import placed after module-level setup on purpose (E402 is not in this repo's rule set, so the directive that used to sit here suppressed nothing)
        if getattr(_pr, "entrypoint_reseeds", None):
            _pr.entrypoint_reseeds = [_thinc_clamped_fix_random_seed if r is _thinc_original_fix else r for r in _pr.entrypoint_reseeds]
    except Exception:  # pragma: no cover -- best-effort pytest-randomly plugin patch; a failure here just skips the shim
        pass
except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover
    warnings.warn(
        f"Skipping thinc pytest-randomly seed shim (thinc import failed: {exc})",
        RuntimeWarning,
        stacklevel=2,
    )

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
    # test_api_stability.py checks ``--refresh-api-snapshot`` in sys.argv to
    # rewrite _api_snapshot.json on an intentional public-API change. Without
    # this addoption pytest rejected the flag as unrecognized, so the documented
    # refresh path was broken -- register it here.
    parser.addoption(
        "--refresh-api-snapshot",
        action="store_true",
        default=False,
        help="rewrite tests/test_meta/_api_snapshot.json instead of comparing (intentional API change)",
    )
    # Several meta-tests support a ``--refresh-<name>-baseline`` flag (checked via sys.argv) to
    # rewrite their committed baseline JSON on an intentional change. Every such flag must be
    # registered here or pytest rejects it as an unrecognized argument (the documented refresh
    # path is otherwise broken).
    for _flag in (
        "--refresh-logger-baseline",
        "--refresh-annotation-baseline",
        "--refresh-docstring-baseline",
        "--refresh-bare-except-baseline",
        "--refresh-console-unicode-baseline",
        "--refresh-debt-baseline",
        "--refresh-mutable-defaults-baseline",
        "--refresh-resource-handle-baseline",
        "--refresh-lock-discipline-baseline",
        "--refresh-param-naming-baseline",
        "--refresh-complexity-baseline",
        "--refresh-unused-param-baseline",
    ):
        parser.addoption(_flag, action="store_true", default=False, help=f"rewrite the corresponding meta-test baseline ({_flag})")
    try:
        # py-ci-shared itself requires python>=3.9 (see the dev-dependency marker in
        # pyproject.toml), so it's absent on the 3.8 CI leg -- without this guard the
        # import breaks conftest loading for the WHOLE test suite on that leg, not just
        # the code-audit test.
        from py_ci_shared.code_audit_meta import register_refresh_option

        register_refresh_option(parser)  # --refresh-code-audit-baseline, shared with every other consumer

        from py_ci_shared.loc_budget import register_refresh_option as register_loc_budget_refresh_option

        register_loc_budget_refresh_option(parser)  # --refresh-loc-budget-baseline
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live LLM-provider test -- pass --run-live to execute")
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
        except Exception:  # best-effort secret-store lookup; falls through to skip below either way
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
    return pd.DataFrame({"int_col": [1, 2, 3, 4, 5], "float_col": [1.1, 2.2, 3.3, 4.4, 5.5], "str_col": ["a", "b", "c", "d", "e"]})


# Removed 2026-09-02: the `mixed_types_df`, `temp_dir`, `float_with_integers_df` and
# `constant_columns_df` fixtures were requested by no test anywhere under tests/ (only
# `sample_df` above is, by test_pandaslib.py). They read as "these DataFrame shapes are covered
# somewhere" when nothing exercised them, and `temp_dir` duplicated pytest's builtin `tmp_path`
# while inviting new tests to use a hand-rolled fixture instead. New tests needing a temporary
# directory should use `tmp_path`; new DataFrame shapes belong next to the tests that use them.
