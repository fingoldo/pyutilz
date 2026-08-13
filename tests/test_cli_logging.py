"""Tests for pyutilz.system.cli_logging."""

from __future__ import annotations

import io
import logging
from unittest.mock import patch

from pyutilz.system.cli_logging import DEFAULT_FORMAT, setup_cli_logging


def test_default_call_configures_root_logger_at_info():
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging()
    mocked.assert_called_once_with(level=logging.INFO, format=DEFAULT_FORMAT, force=True)


def test_custom_level_and_format_are_forwarded():
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging(level=logging.DEBUG, fmt="%(message)s")
    mocked.assert_called_once_with(level=logging.DEBUG, format="%(message)s", force=True)


def test_stream_only_forwarded_when_explicitly_given():
    stream = io.StringIO()
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging(stream=stream)
    mocked.assert_called_once_with(level=logging.INFO, format=DEFAULT_FORMAT, force=True, stream=stream)


def test_force_false_forwards_force_kwarg_as_false():
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging(force=False)
    mocked.assert_called_once_with(level=logging.INFO, format=DEFAULT_FORMAT, force=False)


def test_actually_emits_a_log_line_via_basicConfig(monkeypatch):
    """End-to-end: not just that basicConfig was called with the right kwargs, but that the
    resulting root logger actually emits at the configured level/format."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)  # basicConfig is a no-op if handlers already exist -- reset

    stream = io.StringIO()
    setup_cli_logging(fmt="%(message)s", stream=stream)
    logging.getLogger("pyutilz.test_cli_logging").info("hello")
    assert "hello" in stream.getvalue()

    for h in list(root.handlers):
        root.removeHandler(h)


def test_survives_a_root_logger_already_configured_by_earlier_code(monkeypatch):
    """Regression for the class of bug found in autopsia's integrity.py: plain
    ``logging.basicConfig()`` silently no-ops when the root logger already has a handler
    installed by earlier code in the same process (e.g. an earlier test, or pytest's own
    capturing setup) -- the caller gets no error and no log output. Simulate that leaked state
    (a stray handler pointed at a stream nobody is reading) WITHOUT manually resetting it before
    calling ``setup_cli_logging`` -- the function's own ``force=True`` default must still make the
    call land on the stream this call actually asked for."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        stray = logging.StreamHandler(io.StringIO())
        root.addHandler(stray)
        root.setLevel(logging.WARNING)

        stream = io.StringIO()
        setup_cli_logging(fmt="%(message)s", stream=stream)
        logging.getLogger("pyutilz.test_cli_logging").info("hello")

        assert "hello" in stream.getvalue()
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
