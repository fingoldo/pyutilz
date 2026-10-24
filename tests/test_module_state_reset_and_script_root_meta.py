import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_prove_meta_checks_repo_is_derived_from_file():
    spec = importlib.util.spec_from_file_location("prove_meta_checks_probe", _ROOT / "scripts" / "prove_meta_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.REPO == _ROOT
