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


class TestExactRequestsAndMessages:
    """Survivors of the mutation sweep: the exact headers, URLs, status handling and log texts of the client."""

    @staticmethod
    def _resp(status, text="", payload=None):
        r = MagicMock()
        r.status_code = status
        r.text = text
        r.json.return_value = payload if payload is not None else {}
        return r

    def test_the_basic_auth_header_and_content_type_are_sent_exactly(self):
        import base64

        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = self._resp(HTTPStatus.OK, payload={"response": {"token": "T"}})
            fm.get_session_token(username="user", password="pw")  # pragma: allowlist secret
        expected = "Basic " + base64.b64encode(b"user:pw").decode()
        first = web.connect.call_args_list[0]
        assert first.kwargs["m_template_headers"] == {"Authorization": expected, "Content-Type": "application/json"}

    def test_the_session_request_posts_to_the_sessions_endpoint_without_a_random_agent(self):
        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = self._resp(HTTPStatus.OK, payload={"response": {"token": "T"}})
            fm.get_session_token(username="u", password="p")
        web.get_url.assert_called_once_with("https://fm.example/sessions", b_random_ua=False, verb="post")

    def test_a_token_is_then_sent_as_a_bearer_header(self):
        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = self._resp(HTTPStatus.OK, payload={"response": {"token": "TOK"}})
            assert fm.get_session_token(username="u", password="p") == "TOK"
        assert web.connect.call_args_list[-1].kwargs["m_template_headers"] == {
            "Authorization": "Bearer TOK",
            "Content-Type": "application/json",
        }

    def test_the_default_budget_is_ten_attempts_ten_seconds_apart(self):
        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep") as sleep:
            web.get_url.return_value = None
            assert fm.get_session_token(username="u", password="p") is None
        assert web.get_url.call_count == 10
        assert sleep.call_args_list == [((10,),)] * 9

    def test_a_missing_response_logs_its_exact_warning(self, caplog):
        import logging

        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = None
            with caplog.at_level(logging.WARNING, logger=fm.logger.name):
                fm.get_session_token(username="u", password="p", max_retries=1)
        assert [r.getMessage() for r in caplog.records] == ["No response while getting filemaker session token"]

    def test_an_error_status_logs_the_code_and_a_200_char_prefix_of_the_body(self, caplog):
        import logging

        fm.filemaker_url = "https://fm.example"
        body = "x" * 300
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = self._resp(HTTPStatus.UNAUTHORIZED, text=body)
            with caplog.at_level(logging.WARNING, logger=fm.logger.name):
                fm.get_session_token(username="u", password="p", max_retries=1)
        assert [r.getMessage() for r in caplog.records] == [f"Error 401 while getting filemaker session token: {'x' * 200}"]

    def test_an_invalid_token_logs_only_the_response_keys_sorted(self, caplog):
        import logging

        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = self._resp(HTTPStatus.OK, payload={"zeta": 1, "alpha": 2})
            with caplog.at_level(logging.WARNING, logger=fm.logger.name):
                assert fm.get_session_token(username="u", password="p", max_retries=1) is None
        assert [r.getMessage() for r in caplog.records] == ["Empty/invalid filemaker session token; response keys=['alpha', 'zeta']"]


class TestInitSchemeAndErrors:
    def test_plain_http_is_refused_with_the_exact_message(self):
        with pytest.raises(ValueError) as info:
            fm.init("http://fm.example", "u", "p")
        assert str(info.value) == ("filemaker init: refusing non-https URL (scheme 'http'); credentials would be sent in clear text")
        assert fm.filemaker_url is None

    def test_an_uppercase_https_scheme_is_accepted(self):
        with patch.object(fm, "get_session_token", return_value="T"):
            fm.init("HTTPS://fm.example", "u", "p")
        assert fm.filemaker_url == "HTTPS://fm.example"

    def test_plain_http_is_accepted_only_when_explicitly_allowed(self):
        with patch.object(fm, "get_session_token", return_value="T"):
            fm.init("http://fm.example", "u", "p", allow_insecure_http=True)
        assert fm.filemaker_url == "http://fm.example"

    def test_a_failed_init_raises_with_the_exact_message(self):
        with patch.object(fm, "get_session_token", return_value=None):
            with pytest.raises(RuntimeError) as info:
                fm.init("https://fm.example", "u", "p")
        assert str(info.value) == "filemaker init: could not obtain a session token from https://fm.example -- not initialized."


class TestPostRecord:
    @staticmethod
    def _resp(status, text=""):
        r = MagicMock()
        r.status_code = status
        r.text = text
        return r

    def test_a_successful_post_goes_to_the_layout_records_endpoint_with_the_json_body(self):
        with patch.object(fm, "web") as web:
            web.get_url.return_value = self._resp(HTTPStatus.OK)
            assert fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1}) is True
        web.get_url.assert_called_once_with("https://fm.example/layouts/Orders/records/", verb="post", json={"a": 1})

    def test_the_default_is_three_attempts(self):
        with patch.object(fm, "web") as web:
            web.get_url.return_value = None
            with pytest.raises(ValueError) as info:
                fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1})
        assert web.get_url.call_count == 3
        assert str(info.value) == "Filemaker insert failed after 3 attempts: web.get_url() returned no response"

    def test_a_non_ok_status_raises_with_its_code_and_body(self):
        with patch.object(fm, "web") as web:
            web.get_url.return_value = self._resp(HTTPStatus.INTERNAL_SERVER_ERROR, "boom")
            with pytest.raises(ValueError) as info:
                fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1})
        assert str(info.value) == f"Filemaker insert failed with status {HTTPStatus.INTERNAL_SERVER_ERROR}: boom"


class TestPostLogsAndTokenRefresh:
    @staticmethod
    def _resp(status, text=""):
        r = MagicMock()
        r.status_code = status
        r.text = text
        return r

    def test_an_exception_during_the_post_is_logged_with_the_error_and_the_payload(self, caplog):
        import logging

        with patch.object(fm, "web") as web:
            web.get_url.side_effect = [RuntimeError("socket reset"), self._resp(HTTPStatus.OK)]
            with caplog.at_level(logging.ERROR, logger=fm.logger.name):
                assert fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1}) is True
        assert [r.getMessage() for r in caplog.records] == ["Exception socket reset when inserting into filemaker object {'a': 1}"]

    def test_a_missing_response_is_logged_with_the_payload(self, caplog):
        import logging

        with patch.object(fm, "web") as web:
            web.get_url.side_effect = [None, self._resp(HTTPStatus.OK)]
            with caplog.at_level(logging.ERROR, logger=fm.logger.name):
                assert fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1}) is True
        assert [r.getMessage() for r in caplog.records] == ["No response when inserting into filemaker object {'a': 1}"]

    def test_an_expired_token_is_refreshed_with_its_exact_log_line(self, caplog):
        import logging

        fm.filemaker_url = "https://fm.example"
        fm.filemaker_username, fm.filemaker_password = "u", "p"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.side_effect = [
                self._resp(HTTPStatus.UNAUTHORIZED, "Invalid FileMaker Data API token"),
                self._resp(HTTPStatus.OK),
            ]
            with patch.object(fm, "get_session_token", return_value="NEW") as refresh:
                with caplog.at_level(logging.WARNING, logger=fm.logger.name):
                    assert fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1}) is True
        refresh.assert_called_once_with(username="u", password="p")
        assert [r.getMessage() for r in caplog.records] == ["Refreshing the filemaker token..."]

    def test_a_401_with_another_body_is_not_treated_as_an_expired_token(self):
        with patch.object(fm, "web") as web:
            web.get_url.return_value = self._resp(HTTPStatus.UNAUTHORIZED, "wrong password")
            with patch.object(fm, "get_session_token") as refresh:
                with pytest.raises(ValueError, match="status 401"):
                    fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1})
        refresh.assert_not_called()

    def test_a_non_ok_status_is_logged_with_the_status_and_body(self, caplog):
        import logging

        with patch.object(fm, "web") as web:
            web.get_url.return_value = self._resp(HTTPStatus.BAD_REQUEST, "bad field")
            with caplog.at_level(logging.ERROR, logger=fm.logger.name):
                with pytest.raises(ValueError):
                    fm.post_filemaker_record("https://fm.example", "Orders", {"a": 1})
        assert [r.getMessage() for r in caplog.records] == [f"Exception {HTTPStatus.BAD_REQUEST}: bad field when inserting object details {{'a': 1}}"]


class TestCredentialAndTokenChecks:
    def test_a_missing_password_alone_is_refused_even_when_the_username_is_set(self):
        fm.filemaker_url = "https://fm.example"
        with pytest.raises(ValueError, match="not configured"):
            fm.get_session_token(username="u", password=None)

    def test_a_missing_url_alone_is_refused_even_when_credentials_are_set(self):
        with pytest.raises(ValueError, match="not configured"):
            fm.get_session_token(username="u", password="p")

    def test_an_empty_string_token_is_invalid_and_retried(self):
        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = MagicMock(status_code=HTTPStatus.OK, json=MagicMock(return_value={"response": {"token": ""}}))
            assert fm.get_session_token(username="u", password="p", max_retries=2) is None
        assert web.get_url.call_count == 2

    def test_a_non_string_token_is_invalid_and_retried(self):
        fm.filemaker_url = "https://fm.example"
        with patch.object(fm, "web") as web, patch.object(fm, "sleep"):
            web.get_url.return_value = MagicMock(status_code=HTTPStatus.OK, json=MagicMock(return_value={"response": {"token": 123}}))
            assert fm.get_session_token(username="u", password="p", max_retries=2) is None
        assert web.get_url.call_count == 2
