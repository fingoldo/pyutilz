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
        import dash_bootstrap_components as dbc
        assert isinstance(result, dbc.Container)
        assert len(result.children) == 2  # header (dbc.Tabs) + body (html.Div), no tooltips

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


# ── 2026-08-25: loading overlay on tab content ──────────────────────────────


def _tabs(**kwargs):
    from pyutilz.dev.dashlib import create_tabs

    return create_tabs(
        tabsName="T",
        tabsList=[["One", "One", None], ["Two", "Two", None]],
        draw_tab_content_function=lambda tab: f"body-{tab}",
        **kwargs,
    )


def _find(node, predicate):
    """Depth-first search over a Dash component tree."""
    if predicate(node):
        return node
    children = getattr(node, "children", None)
    if children is None:
        return None
    for child in children if isinstance(children, (list, tuple)) else [children]:
        found = _find(child, predicate)
        if found is not None:
            return found
    return None


class TestTabContentLoadingOverlay:
    """The tab HEADER switches the instant it is clicked -- that is client-side state -- while the
    body waits on a callback, which can be seconds against a large database. Without an overlay the
    user sees the new tab's header above the PREVIOUS tab's content, which reads as finished rather
    than pending.
    """

    def test_content_is_wrapped_by_default(self, app_context):
        from dash import dcc

        tabs = _tabs()
        loading = _find(tabs, lambda n: isinstance(n, dcc.Loading))
        assert loading is not None, "tab content should carry a loading overlay out of the box"
        # The overlay must WRAP the div the callback replaces -- wrapping the returned content
        # instead would only appear once the wait was already over.
        inner = loading.children
        assert getattr(inner, "id", None) == "tabsTContent"

    def test_can_be_opted_out(self, app_context):
        from dash import dcc

        assert _find(_tabs(show_loading=False), lambda n: isinstance(n, dcc.Loading)) is None

    def test_content_div_keeps_its_id_either_way(self, app_context):
        """The id is the callback's Output; wrapping must not move or rename it."""
        for kwargs in ({}, {"show_loading": False}):
            tabs = _tabs(**kwargs)
            assert _find(tabs, lambda n: getattr(n, "id", None) == "tabsTContent") is not None, kwargs

    def test_spinner_is_delayed_so_fast_tabs_do_not_flash(self, app_context):
        from dash import dcc

        loading = _find(_tabs(), lambda n: isinstance(n, dcc.Loading))
        assert loading.delay_show >= 100
