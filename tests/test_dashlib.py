"""Regression tests for pyutilz.dev.dashlib.create_tabs (2026-07-21 audit fixes).

dev/dashlib.py unconditionally imports flask/dash/dash_bootstrap_components at module level
(documented pyproject.toml [dash]-extra exception) -- gracefully skipped here when that extra
isn't installed, matching the project's established pattern for other heavy-optional-dep test
files (see test_browser_regression.py, test_tokenizers_extra.py).
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

pytest.importorskip("dash_bootstrap_components")
pytest.importorskip("flask")

from flask import Flask

from pyutilz.dev.dashlib import create_tabs


@pytest.fixture
def app_context():
    app = Flask(__name__)
    app.secret_key = "test"  # pragma: allowlist secret
    with app.test_request_context():
        yield


def _draw_content(active_tab):
    return f"content for {active_tab}"


class TestCreateTabsNoFlaskLogin:
    """flask_login absent/erroring (e.g. not installed, or no active session context) --
    create_tabs must still render tabs (regression test for the NameError bug: `user` was
    previously only bound inside the try block, so this exact code path -- which the
    surrounding except is explicitly documented to tolerate -- left `user` unbound and crashed
    the moment a role-restricted tab was reached). Simulated by making `current_user.is_authenticated`
    itself raise, exercising exactly the code path the try/except is designed to catch,
    regardless of whether flask_login happens to be installed in the environment running this
    test."""

    @staticmethod
    def _broken_current_user():
        broken = MagicMock()
        type(broken).is_authenticated = PropertyMock(side_effect=RuntimeError("no request/session context"))
        return broken

    def test_unrestricted_tabs_render_without_flask_login(self, app_context):
        with patch("flask_login.current_user", self._broken_current_user()):
            tabsList = [("Home", "home", None), ("About", "about", None)]
            result = create_tabs("mytabs", tabsList, _draw_content)
        assert result is not None

    def test_role_restricted_tab_is_skipped_not_crashed(self, app_context):
        # Previously: NameError: name 'user' is not defined, since the try block's exception
        # left `user` unbound before role-restricted tabs were evaluated.
        with patch("flask_login.current_user", self._broken_current_user()):
            tabsList = [("Home", "home", None), ("Admin", "admin", ["admin"])]
            result = create_tabs("mytabs", tabsList, _draw_content)
        assert result is not None
        # Only the unrestricted "Home" tab should have rendered.
        tabs_component = result.children[0]
        assert len(tabs_component.children) == 1


class TestCreateTabsWithFlaskLogin:
    def test_authenticated_user_role_match_renders_tab(self, app_context):
        fake_user = MagicMock(is_authenticated=True, role="admin")
        with patch("flask_login.current_user", fake_user):
            tabsList = [("Admin", "admin", ["admin"])]
            result = create_tabs("mytabs", tabsList, _draw_content)
        assert result is not None
        tabs_component = result.children[0]
        assert len(tabs_component.children) == 1

    def test_authenticated_user_role_mismatch_skips_tab(self, app_context):
        fake_user = MagicMock(is_authenticated=True, role="viewer")
        with patch("flask_login.current_user", fake_user):
            tabsList = [("Admin", "admin", ["admin"])]
            result = create_tabs("mytabs", tabsList, _draw_content)
        # No tabs matched -> create_tabs returns None (len(tabs) == 0 branch).
        assert result is None

    def test_unauthenticated_user_returns_none(self, app_context):
        fake_user = MagicMock(is_authenticated=False)
        with patch("flask_login.current_user", fake_user):
            tabsList = [("Home", "home", None)]
            result = create_tabs("mytabs", tabsList, _draw_content)
        assert result is None


class TestCreateTabsTooltip:
    """Regression test: the 6th tuple element (tabTooltip) was computed via a bare, unassigned
    expression statement (`tabClassNames[2]`) and never wired into a dbc.Tooltip -- the feature
    was completely non-functional despite being documented in the function's own docstring."""

    def test_tooltip_is_rendered_when_provided(self, app_context):
        tabsList = [("Home", "home", None, "cls1", "lbl1", "This is a tooltip")]
        result = create_tabs("mytabs", tabsList, _draw_content)
        assert result is not None
        # data = [header, body, *tooltips] per the fix; a Tooltip component should be present.
        tooltip_children = [c for c in result.children if type(c).__name__ == "Tooltip"]
        assert len(tooltip_children) == 1
        assert tooltip_children[0].children == "This is a tooltip"
        assert tooltip_children[0].target == "tabhome"

    def test_no_tooltip_when_not_provided(self, app_context):
        tabsList = [("Home", "home", None)]
        result = create_tabs("mytabs", tabsList, _draw_content)
        tooltip_children = [c for c in result.children if type(c).__name__ == "Tooltip"]
        assert len(tooltip_children) == 0
