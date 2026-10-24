from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_f07_beachhead_includes_the_two_new_subpackages():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pyutilz.web.proxy.*"' in text
    assert '"pyutilz.llm.openrouter_provider.*"' in text
