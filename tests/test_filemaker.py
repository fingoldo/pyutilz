"""Behavioral coverage for pyutilz.core.filemaker, previously only import-smoke-tested
(test_smoke_untested_modules.py checks the module imports and exposes get_session_token/init,
but never exercises any actual behavior)."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

import pyutilz.core.filemaker as fm


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset the module-level credential globals before and after each test so tests don't
    bleed state through the shared module (filemaker_url/username/password are set by init())."""
    fm.filemaker_url, fm.filemaker_username, fm.filemaker_password = None, None, None
    yield
    fm.filemaker_url, fm.filemaker_username, fm.filemaker_password = None, None, None


class TestSimplifyTypes:
    """simplify_types -- pure dict transform, no network."""

    def test_list_value_joined_with_separator(self):
        out = fm.simplify_types({"a": [1, 2, 3]})
        assert out["a"] == "1,2,3"

    def test_list_value_custom_separator(self):
        out = fm.simplify_types({"a": [1, 2]}, sep="|")
        assert out["a"] == "1|2"

    def test_dict_value_stringified(self):
        out = fm.simplify_types({"a": {"x": 1}})
        assert out["a"] == str({"x": 1})

    def test_none_value_dropped(self):
        out = fm.simplify_types({"a": None, "b": 1})
        assert "a" not in out
        assert out["b"] == 1

    def test_scalar_values_untouched(self):
        out = fm.simplify_types({"a": 1, "b": "text", "c": 2.5})
        assert out == {"a": 1, "b": "text", "c": 2.5}

    def test_returns_a_new_dict_and_leaves_the_argument_untouched(self):
        """simplify_types is a transform, not an in-place mutator (2026-09-03 audit, F133).

        It used to mutate the caller's dict -- ``obj.copy()`` copied only the iteration view -- so
        a caller that still needed its original list fields, or its None-valued keys, silently lost
        them. The returned dict carries the simplified values; the input is unchanged."""
        obj = {"a": [1, 2], "b": None}
        out = fm.simplify_types(obj)
        assert out is not obj
        assert out["a"] == "1,2"
        assert obj == {"a": [1, 2], "b": None}


class TestGetSessionToken:
    """get_session_token."""

    def test_raises_without_configured_credentials(self):
        """No init() call and no explicit username/password/url -- must raise, not silently proceed."""
        with pytest.raises(ValueError, match="not configured"):
            fm.get_session_token()

    def test_explicit_credentials_used_when_module_globals_unset(self):
        """Explicit username/password bypass the module-global requirement, but filemaker_url
        must still be configured (via a prior init() call)."""
        fm.filemaker_url = "https://fm.example.com"
        with patch.object(fm, "web") as mock_web:
            mock_resp = MagicMock(status_code=HTTPStatus.OK)
            mock_resp.json.return_value = {"response": {"token": "tok123"}}
            mock_web.get_url.return_value = mock_resp
            token = fm.get_session_token(username="u", password="p")
        assert token == "tok123"

    def test_successful_token_fetch_returns_token(self):
        fm.filemaker_url = "https://fm.example.com"
        fm.filemaker_username = "u"
        fm.filemaker_password = "p"
        with patch.object(fm, "web") as mock_web:
            mock_resp = MagicMock(status_code=HTTPStatus.OK)
            mock_resp.json.return_value = {"response": {"token": "abc"}}
            mock_web.get_url.return_value = mock_resp
            token = fm.get_session_token()
        assert token == "abc"
        # Second web.connect call re-authenticates with the bearer token.
        bearer_calls = [c for c in mock_web.connect.call_args_list if "Bearer" in str(c)]
        assert len(bearer_calls) == 1

    def test_none_response_retries_then_gives_up(self):
        """web.get_url returning None on every attempt exhausts retries and returns None."""
        fm.filemaker_url = "https://fm.example.com"
        fm.filemaker_username = "u"
        fm.filemaker_password = "p"
        with patch.object(fm, "web") as mock_web, patch.object(fm, "sleep") as mock_sleep:
            mock_web.get_url.return_value = None
            token = fm.get_session_token(max_retries=2, sleep_int_seconds=0)
        assert token is None
        assert mock_web.get_url.call_count == 2
        # N attempts cost N-1 sleeps: there is nothing to wait for after the final one
        # (2026-09-03 audit, F110).
        assert mock_sleep.call_count == 1

    def test_non_ok_status_retries_then_gives_up(self):
        fm.filemaker_url = "https://fm.example.com"
        fm.filemaker_username = "u"
        fm.filemaker_password = "p"
        with patch.object(fm, "web") as mock_web, patch.object(fm, "sleep"):
            mock_resp = MagicMock(status_code=HTTPStatus.UNAUTHORIZED, text="denied")
            mock_web.get_url.return_value = mock_resp
            token = fm.get_session_token(max_retries=1, sleep_int_seconds=0)
        assert token is None

    def test_empty_token_in_response_retries_then_gives_up(self):
        """A 200 response with no usable token string in it must not be returned as a truthy token."""
        fm.filemaker_url = "https://fm.example.com"
        fm.filemaker_username = "u"
        fm.filemaker_password = "p"
        with patch.object(fm, "web") as mock_web, patch.object(fm, "sleep"):
            mock_resp = MagicMock(status_code=HTTPStatus.OK)
            mock_resp.json.return_value = {"response": {}}
            mock_web.get_url.return_value = mock_resp
            token = fm.get_session_token(max_retries=1, sleep_int_seconds=0)
        assert token is None


class TestInit:
    """init."""

    def test_init_stores_credentials_and_authenticates(self):
        with patch.object(fm, "web") as mock_web:
            mock_resp = MagicMock(status_code=HTTPStatus.OK)
            mock_resp.json.return_value = {"response": {"token": "tok"}}
            mock_web.get_url.return_value = mock_resp
            fm.init("https://fm.example.com", "user", "pass")
        assert fm.filemaker_url == "https://fm.example.com"
        assert fm.filemaker_username == "user"
        assert fm.filemaker_password == "pass"  # pragma: allowlist secret -- test placeholder, not a real credential


class TestPostFilemakerRecord:
    """post_filemaker_record."""

    def test_successful_post_returns_true(self):
        with patch.object(fm, "web") as mock_web:
            mock_web.get_url.return_value = MagicMock(status_code=HTTPStatus.OK)
            out = fm.post_filemaker_record("https://fm.example.com", "Layout1", {"field": "value"})
        assert out is True

    def test_persistent_failure_raises_after_attempts_exhausted(self):
        with patch.object(fm, "web") as mock_web:
            mock_web.get_url.side_effect = RuntimeError("boom")
            with pytest.raises(ValueError, match="Filemaker insert failed"):
                fm.post_filemaker_record("https://fm.example.com", "Layout1", {"field": "value"}, num_attempts=2)
        assert mock_web.get_url.call_count == 2

    def test_none_response_counts_as_failure_and_continues_retrying(self):
        with patch.object(fm, "web") as mock_web:
            mock_web.get_url.return_value = None
            with pytest.raises(ValueError, match="Filemaker insert failed"):
                fm.post_filemaker_record("https://fm.example.com", "Layout1", {"field": "value"}, num_attempts=2)
        assert mock_web.get_url.call_count == 2

    def test_non_401_error_status_raises_immediately_without_retry(self):
        """A non-401 error status is a hard failure -- no point retrying, raises right away."""
        with patch.object(fm, "web") as mock_web:
            mock_web.get_url.return_value = MagicMock(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, text="oops")
            with pytest.raises(ValueError, match="status 500"):
                fm.post_filemaker_record("https://fm.example.com", "Layout1", {"field": "value"}, num_attempts=3)
        # Raises on the FIRST attempt -- no retry loop for a non-token error.
        assert mock_web.get_url.call_count == 1

    def test_expired_token_refreshes_and_raises_if_refresh_fails(self):
        """A 401 'Invalid FileMaker Data API token' triggers a refresh attempt; if that refresh
        itself fails (no token returned), a ValueError names the refresh failure specifically."""
        fm.filemaker_username = "u"
        fm.filemaker_password = "p"
        with patch.object(fm, "web") as mock_web, patch.object(fm, "get_session_token", return_value=None) as mock_refresh:
            mock_web.get_url.return_value = MagicMock(status_code=401, text="Invalid FileMaker Data API token")
            with pytest.raises(ValueError, match="Could not refresh"):
                fm.post_filemaker_record("https://fm.example.com", "Layout1", {"field": "value"}, num_attempts=1)
        mock_refresh.assert_called_once()
