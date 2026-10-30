"""Behaviour of ``pyutilz.system.system.memory``: the own-process memory reading is a non-negative number or None, the Windows
trim is a no-op returning a bool on other platforms, and the session-object listing returns a frame ranked by size. The
trim and the listing are exercised only through their return shape."""

from __future__ import annotations

import sys

from pyutilz.system.system import memory


def test_the_own_memory_reading_is_non_negative_or_unknown():
    reading = memory.get_own_memory_usage()
    assert reading is None or reading >= 0


def test_the_windows_trim_returns_a_bool():
    assert isinstance(memory.trim_windows_process_memory(), bool)


def test_on_other_platforms_the_trim_reports_no_work():
    if sys.platform.startswith("win"):
        return
    assert memory.trim_windows_process_memory() is False


def test_the_biggest_session_objects_are_a_frame_with_one_row_per_object():
    session = {"small": [1], "big": list(range(10_000)), "text": "x" * 5_000}
    frame = memory.show_biggest_session_objects(session, N=2, min_size_bytes=1)
    assert len(frame) == 2


def test_the_trim_returns_false_without_touching_windows_apis_on_linux(monkeypatch):
    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    assert memory.trim_windows_process_memory() is False
