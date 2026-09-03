"""Regression tests for the 2026-09-03 audit findings in pyutilz.dev.dashlib.

Carved out of test_domain_core_dev_system_audit_20260903.py, which sits at the 1000-LOC
budget: these share one precondition (the [dash] extra) and belong together.
"""

import pytest


def _dashlib():
    """The dashlib module, or a skip: it needs the [dash] extra, which CI does not install."""
    pytest.importorskip("flask")
    from pyutilz.dev import dashlib

    return dashlib


def test_f55_active_tab_is_seeded_from_the_tab_id(monkeypatch):
    """tabsList[0][0] is the LABEL; ids come from [1]. The seeded value matched no real tab, so no
    tab rendered selected and the content callback ran against a nonexistent id."""
    dashlib = _dashlib()

    session: dict = {}
    monkeypatch.setattr(dashlib, "session", session)
    dashlib.create_tabs("Main", [("Overview", "ov", None), ("Details", "dt", None)], lambda tab_id: "x")
    assert session["tabsMainActiveTab"] == "tabov"


def test_f113_empty_tabs_list_returns_none_instead_of_raising(monkeypatch):
    dashlib = _dashlib()

    monkeypatch.setattr(dashlib, "session", {})
    assert dashlib.create_tabs("Main", [], lambda tab_id: "x") is None


def test_f145_missing_label_class_name_is_not_rendered_as_the_string_none(monkeypatch):
    dashlib = _dashlib()

    monkeypatch.setattr(dashlib, "session", {})
    container = dashlib.create_tabs("Main", [("Overview", "ov", None)], lambda tab_id: "x")
    assert "None" not in str(container)
