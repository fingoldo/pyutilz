"""Tests for pyutilz.system.cli_logging."""

from __future__ import annotations

import io
import logging
from unittest.mock import patch

from pyutilz.system.cli_logging import DEFAULT_FORMAT, setup_cli_logging


def test_default_call_configures_root_logger_at_info():
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging()
    mocked.assert_called_once_with(level=logging.INFO, format=DEFAULT_FORMAT)


def test_custom_level_and_format_are_forwarded():
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging(level=logging.DEBUG, fmt="%(message)s")
    mocked.assert_called_once_with(level=logging.DEBUG, format="%(message)s")


def test_stream_only_forwarded_when_explicitly_given():
    stream = io.StringIO()
    with patch("logging.basicConfig") as mocked:
        setup_cli_logging(stream=stream)
    mocked.assert_called_once_with(level=logging.INFO, format=DEFAULT_FORMAT, stream=stream)


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
