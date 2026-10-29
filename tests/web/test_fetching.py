"""Behaviour of ``pyutilz.web.web.fetching._prepare_headers``: a custom header set replaces the session's base set, injected
headers are added on top without mutating the caller's dict, and sorting and lowercasing are applied only when asked. The
network is not touched."""

from __future__ import annotations

from pyutilz.web.web import fetching


def test_custom_headers_win_over_the_base_set():
    assert fetching._prepare_headers({"A": "1"}, {"B": "2"}, None, False, False) == {"A": "1"}


def test_the_base_set_is_used_when_no_custom_set_is_given():
    assert fetching._prepare_headers(None, {"B": "2"}, None, False, False) == {"B": "2"}


def test_injected_headers_are_added_without_mutating_the_base_set():
    base = {"B": "2"}
    out = fetching._prepare_headers(None, base, {"C": "3"}, False, False)
    assert out == {"B": "2", "C": "3"}
    assert base == {"B": "2"}


def test_headers_are_sorted_and_lowercased_when_asked():
    out = fetching._prepare_headers({"Zeta": "z", "Alpha": "a"}, None, None, True, True)
    assert list(out.items()) == [("alpha", "a"), ("zeta", "z")]


def test_no_headers_at_all_gives_none_or_empty():
    assert not fetching._prepare_headers(None, None, None, False, False)
