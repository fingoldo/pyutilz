"""Behaviour of ``pyutilz.system.system.sysinfo._decode_child_output``: child-process bytes decode as UTF-8 text, and a byte
that is not valid is replaced rather than raising, so a stray byte in a probe's output never aborts the system report."""

from __future__ import annotations

import platform

from pyutilz.system.system import sysinfo


def test_bytes_decode_with_the_platform_console_encoding():
    raw = "héllo".encode()
    expected = raw.decode("oem", errors="replace") if platform.system() == "Windows" else raw.decode("utf-8")
    assert sysinfo._decode_child_output(raw) == expected


def test_a_plain_ascii_payload_decodes_exactly():
    assert sysinfo._decode_child_output(b"nvidia-smi 535.0") == "nvidia-smi 535.0"


def test_an_invalid_byte_is_replaced_not_raised():
    decoded = sysinfo._decode_child_output(b"ok\xff\xfe")
    assert decoded.startswith("ok")
