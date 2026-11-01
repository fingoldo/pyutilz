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


class TestSecondRoundPatterns:
    def test_an_empty_password_in_a_dsn_is_left_alone(self):
        assert redact_secrets("postgresql://user:@host/db") == "postgresql://user:@host/db"  # pragma: allowlist secret

    def test_an_empty_password_query_parameter_is_left_alone(self):
        assert sanitize_dsn("postgresql://host/db?password=&sslmode=require") == "postgresql://host/db?password=&sslmode=require"  # pragma: allowlist secret

    def test_a_dsn_scheme_must_start_at_a_word_boundary(self):
        assert redact_secrets("xpostgres://user:pw@host/db") == "xpostgres://user:***@host/db"  # pragma: allowlist secret

    def test_a_token_query_parameter_is_masked(self):
        assert redact_secrets("https://x.example/?token=abc123&x=1") == "https://x.example/?***&x=1"

    def test_an_api_key_query_parameter_is_masked(self):
        assert redact_secrets("https://x.example/?api_key=abc123&x=1") == "https://x.example/?***&x=1"

    def test_an_api_dash_key_query_parameter_is_masked(self):
        assert redact_secrets("https://x.example/?api-key=abc123&x=1") == "https://x.example/?***&x=1"

    def test_an_authorization_header_keeps_its_scheme_label(self):
        assert redact_secrets("Authorization: Bearer abc123") == "Authorization: Bearer ***"

    def test_an_authorization_header_is_normalised_to_one_space_after_its_colon(self):
        assert redact_secrets("Authorization:   Bearer abc123") == "Authorization: Bearer ***"

    def test_a_query_password_is_masked_in_place_by_the_redactor(self):
        result = redact_secrets("https://h.example/?password=s3cret&x=1")
        assert result.startswith("https://h.example/?password=")
        assert "s3cret" not in result
        assert "***" in result

    def test_a_query_password_is_masked_in_place_by_the_dsn_sanitiser(self):
        assert sanitize_dsn("postgresql://host/db?password=s3cret&sslmode=require") == "postgresql://host/db?password=***&sslmode=require"  # pragma: allowlist secret

    def test_an_authorization_basic_header_keeps_its_scheme_label(self):
        assert redact_secrets("Authorization: Basic dXNlcjpwYXNz") == "Authorization: Basic ***"

    def test_a_bare_bearer_token_keeps_its_label(self):
        assert redact_secrets("send Bearer abc123 now") == "send Bearer *** now"

    def test_a_bare_basic_token_keeps_its_label(self):
        assert redact_secrets("send Basic dXNlcjpwYXNz now") == "send Basic *** now"

    def test_an_aws_access_key_id_is_masked(self):
        assert redact_secrets("key AKIAIOSFODNN7EXAMPLE end") == "key *** end"  # pragma: allowlist secret

    def test_an_openai_style_secret_key_is_masked(self):
        assert redact_secrets("key sk-proj-abcdefghijklmnopqrstuvwx end") == "key *** end"

    def test_a_pem_private_key_block_is_replaced_whole(self):
        block = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj\n-----END RSA PRIVATE KEY-----"  # pragma: allowlist secret
        assert redact_secrets("before " + block + " after") == "before <private-key-redacted> after"

    def test_a_pem_block_without_its_footer_is_not_matched(self):
        text = "-----BEGIN PRIVATE KEY-----\nMIIB"  # pragma: allowlist secret
        assert redact_secrets(text) == text

    def test_a_key_value_secret_keeps_its_key_and_masks_the_value(self):
        assert redact_secrets("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI") == "AWS_SECRET_ACCESS_KEY=***"

    def test_a_credential_name_matches_as_a_whole_word_only(self):
        from pyutilz.text.secrets_scrub import CREDENTIAL_NAME_RE

        assert CREDENTIAL_NAME_RE.search("db_password")
        assert CREDENTIAL_NAME_RE.search("password")
        assert CREDENTIAL_NAME_RE.search("api_key")
        assert not CREDENTIAL_NAME_RE.search("mypassword")
        assert not CREDENTIAL_NAME_RE.search("passwords")
        assert not CREDENTIAL_NAME_RE.search("username")
