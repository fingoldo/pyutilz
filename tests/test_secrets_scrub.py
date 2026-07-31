"""Tests for pyutilz.text.secrets_scrub."""

from __future__ import annotations

from pyutilz.text.secrets_scrub import redact_secrets, sanitize_dsn


class TestSanitizeDsn:
    def test_masks_password_preserves_host(self):
        assert sanitize_dsn("postgresql://user:s3cret@host/db") == "postgresql://user:***@host/db"  # pragma: allowlist secret

    def test_password_containing_at_sign_fully_masked(self):
        assert sanitize_dsn("postgresql://user:p@ss@host/db") == "postgresql://user:***@host/db"

    def test_query_param_password_masked(self):
        assert sanitize_dsn("postgresql://host/db?user=u&password=s3cret&sslmode=require") == "postgresql://host/db?user=u&password=***&sslmode=require"

    def test_text_without_dsn_unaffected(self):
        assert sanitize_dsn("OperationalError('connection refused')") == "OperationalError('connection refused')"


class TestRedactSecrets:
    def test_scheme_qualified_dsn_fully_redacted(self):
        assert redact_secrets("postgresql://user:s3cret@host/db") == "<dsn-redacted>"  # pragma: allowlist secret

    def test_dsn_embedded_in_sentence(self):
        assert redact_secrets("connect to postgres://admin:s3cret@db.example.com/prod fast") == "connect to <dsn-redacted> fast"  # pragma: allowlist secret

    def test_bearer_token_redacted_label_preserved(self):
        assert redact_secrets("Send Bearer ghp_secret_TOKEN123 to authenticate.") == "Send Bearer *** to authenticate."  # pragma: allowlist secret

    def test_authorization_bearer_header_redacted(self):
        assert redact_secrets("Authorization: Bearer ghp_secret_TOKEN123") == "Authorization: Bearer ***"  # pragma: allowlist secret

    def test_key_value_secrets_redacted(self):
        assert redact_secrets("call failed: api_key=sk-super-secret-123 token: deadbeefcafe") == "call failed: api_key=*** token=***"  # pragma: allowlist secret

    def test_oauth_prose_not_falsely_redacted(self):
        """A bare space after "auth" (no `:`/`=`) must NOT trigger the generic key=value scrubber."""
        text = "and oauth flows are supported"
        assert redact_secrets(text) == text

    def test_empty_and_none_return_empty_string(self):
        assert redact_secrets("") == ""
        assert redact_secrets(None) == ""

    def test_limit_truncates_result(self):
        assert redact_secrets("plain text with no secrets", limit=5) == "plain"
