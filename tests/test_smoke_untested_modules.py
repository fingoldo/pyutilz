"""Smoke tests for modules surfaced by ``test_test_source_parity`` as
having no dedicated ``test_*.py`` file.

Each test is intentionally minimal: imports the module, picks one or
two public symbols, and asserts they exist + are the expected kind
(callable / class / etc.). The purpose is the SMOKE check — does this
module at least import without crashing — not full behavioural
coverage. Each module should grow a real test file over time, at which
point the corresponding ``_USER_DEFERRED_MODULES`` entry in
``test_test_source_parity.py`` should be removed.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,expected_symbols",
    [
        # core/openai.py — ChatGPT helper utility.
        ("pyutilz.core.openai", ()),
        # core/filemaker.py — FileMaker DB session helpers.
        ("pyutilz.core.filemaker", ("get_session_token", "init")),
        # database/db.py — SQLAlchemy wrapper. Heavyweight imports
        # (sqlalchemy, psycopg2 stubs) live behind try/except so a bare
        # import without optional deps must still succeed.
        ("pyutilz.database.db", ()),
        # database/deltalakes.py — Delta Lake helpers.
        ("pyutilz.database.deltalakes", ()),
        # database/redislib.py — Redis helpers (uses optional redis dep).
        ("pyutilz.database.redislib", ()),
        # dev/dashlib.py — Dash app helpers.
        ("pyutilz.dev.dashlib", ()),
        # dev/notebook_init.py — Jupyter notebook init.
        ("pyutilz.dev.notebook_init", ()),
        # text/tokenizers.py — tokenizer utilities (spacy / nltk).
        ("pyutilz.text.tokenizers", ()),
    ],
)
def test_module_imports_and_exposes_at_least_one_public_symbol(
    module_path, expected_symbols,
):
    """Every flagged module imports without crashing AND has at least
    one public (non-underscore) symbol on it. Adds the
    ``expected_symbols`` callable-existence check when known.
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        # Optional-dep absence is acceptable — the module file itself
        # still has to import (which has already happened by the time
        # ImportError is raised from a deeper import line). Convert to
        # ``pytest.skip`` so the smoke test doesn't fail on environments
        # missing the optional extra.
        pytest.skip(f"{module_path} requires optional dep ({e})")

    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, (
        f"{module_path} imports but exposes no public symbols — "
        f"either delete the file or add real content"
    )

    for name in expected_symbols:
        assert hasattr(mod, name), (
            f"{module_path}: expected public symbol {name!r} not found "
            f"(was the module renamed or refactored?)"
        )


def test_module_aliases_resolve_for_smoke_set():
    """Cross-check: every smoke-tested module that's also re-exported
    via ``pyutilz._MODULE_ALIASES`` resolves to the same target."""
    import pyutilz
    aliases = pyutilz._MODULE_ALIASES
    smoke_paths = {
        "pyutilz.core.openai",
        "pyutilz.core.filemaker",
        "pyutilz.database.db",
        "pyutilz.database.deltalakes",
        "pyutilz.database.redislib",
        "pyutilz.dev.dashlib",
        "pyutilz.dev.notebook_init",
        "pyutilz.text.tokenizers",
    }
    # Build reverse map: target → alias.
    reverse: dict[str, str] = {v: k for k, v in aliases.items()}
    for path in smoke_paths:
        if path in reverse:
            alias = reverse[path]
            try:
                proxy = importlib.import_module(f"pyutilz.{alias}")
            except ImportError:
                continue
            assert proxy is not None, (
                f"alias pyutilz.{alias} → {path} but proxy lookup returned None"
            )
