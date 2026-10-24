from __future__ import annotations
from pathlib import Path
import pytest


def test_s07_existence_check_error_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pyutilz.core.pythonlib import filesystem

    def _denied(path: str) -> bool:
        raise PermissionError("denied")

    monkeypatch.setattr(filesystem, "exists", _denied)
    with pytest.raises(PermissionError):
        filesystem.load_file(str(tmp_path / "model.joblib"))


def test_s07_missing_file_still_raises_file_not_found(tmp_path: Path) -> None:
    from pyutilz.core.pythonlib import filesystem

    with pytest.raises(FileNotFoundError):
        filesystem.load_file(str(tmp_path / "absent.joblib"))
