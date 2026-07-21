"""Meta-test proposed after the 2026-07-21 audit: every retry predicate in this codebase names
specific third-party SDK exception classes (``anthropic.RateLimitError``,
``google.genai.errors.ServerError``, ``redis.exceptions.AuthenticationError``, ...) and, in two of
the three real cases the audit found, got that class hierarchy WRONG -- either naming a class that
doesn't exist in the actually-installed SDK version (google.genai.errors), or missing that a
"permanent failure" class is secretly a SUBCLASS of the "transient, always retry" class it's
sibling to in the same except clause (redis AuthenticationError is a ConnectionError subclass).
Both bugs are invisible from reading the retry predicate's own source -- they only surface by
introspecting the REAL installed SDK's class hierarchy at runtime, exactly what this test does.

Guards every check with ``pytest.importorskip`` -- these are all optional dependencies; a CI run
without one installed skips that provider's check rather than failing on an unrelated missing
package (matches this repo's optional-dependency-isolation convention elsewhere in test_meta/).
"""
from __future__ import annotations

import pytest


def test_anthropic_retry_classes_exist_and_status_code_predicate_holds():
    """anthropic_provider.py's retry predicate names 4 SDK exception classes
    (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) and, for the
    APIStatusError branch, reads ``.status_code`` to decide retryability. If the installed SDK
    ever renames/removes one of these classes, or APIStatusError stops exposing ``status_code``,
    this fails loudly instead of the retry predicate silently going stale (the exact "SDK moved,
    predicate didn't" shape the 2026-07-21 audit found for OverloadedError/ServiceUnavailableError/
    DeadlineExceededError -- separate APIStatusError siblings, not InternalServerError subclasses)."""
    anthropic = pytest.importorskip("anthropic")
    names = ("RateLimitError", "APIConnectionError", "APITimeoutError", "APIStatusError")
    missing = [n for n in names if not hasattr(anthropic, n)]
    assert not missing, f"anthropic_provider.py's retry predicate references class(es) no longer in the installed SDK: {missing}"

    for n in names:
        assert issubclass(getattr(anthropic, n), Exception), f"anthropic.{n} is no longer an Exception subclass"

    # The retry predicate's lambda reads `.status_code` off any APIStatusError instance -- build a
    # real one (via a fake httpx.Response, same as the SDK's own __init__ derives it) to confirm
    # that attribute is still populated, not silently renamed/dropped.
    import httpx

    fake_response = httpx.Response(status_code=529, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    status_err = anthropic.APIStatusError("overloaded", response=fake_response, body=None)
    assert getattr(status_err, "status_code", None) == 529, (
        "anthropic.APIStatusError instances no longer expose a `.status_code` attribute matching "
        "the response status -- anthropic_provider.py's retry predicate reads "
        "`getattr(e, 'status_code', None)` and would silently always get None (never retrying any "
        "APIStatusError) if this drifted."
    )


def test_gemini_retryable_error_classes_exist_in_installed_sdk():
    """gemini_provider.py's ``_is_retryable_genai_error`` names ``google.genai.errors.ClientError``/
    ``ServerError``. The original regression (2026-07-21 audit) was the predicate referencing a
    DIFFERENT library's (google.api_core.exceptions) classes entirely, matching nothing a real
    google-genai call ever raises -- so 429/5xx responses were silently never retried. This
    verifies the classes this module's own ``except ImportError`` guard depends on are real."""
    errors_mod = pytest.importorskip("google.genai.errors")
    assert hasattr(errors_mod, "ClientError"), "google.genai.errors.ClientError no longer exists in the installed SDK"
    assert hasattr(errors_mod, "ServerError"), "google.genai.errors.ServerError no longer exists in the installed SDK"

    from pyutilz.llm.gemini_provider import _is_retryable_genai_error

    server_err = errors_mod.ServerError.__new__(errors_mod.ServerError)
    assert _is_retryable_genai_error(server_err) is True, "ServerError must always be retryable"

    client_err_429 = errors_mod.ClientError.__new__(errors_mod.ClientError)
    client_err_429.code = 429
    assert _is_retryable_genai_error(client_err_429) is True, "ClientError with code=429 (rate limit) must be retryable"

    client_err_400 = errors_mod.ClientError.__new__(errors_mod.ClientError)
    client_err_400.code = 400
    assert _is_retryable_genai_error(client_err_400) is False, "ClientError with code=400 (bad request) must NOT be retryable"


def test_redis_authentication_error_is_a_connection_error_subclass():
    """redislib.py's retry loop relies on this exact, non-obvious fact: redis-py's
    AuthenticationError is a SUBCLASS of ConnectionError -- so a bare ``except ConnectionError:``
    retry branch would catch a permanent wrong-password failure too, looping forever. The fix
    (2026-07-21 audit) added a dedicated ``except AuthenticationError:`` branch placed BEFORE the
    ConnectionError branch specifically because of this subclass relationship. If a future redis-py
    release ever breaks this inheritance, this test fails loudly instead of the ordering-dependent
    fix silently becoming a no-op (dead branch never reached, or -- if inheritance is REMOVED --
    the exclusion redislib.py relies on simply stops applying)."""
    redis_exceptions = pytest.importorskip("redis.exceptions")
    assert issubclass(redis_exceptions.AuthenticationError, redis_exceptions.ConnectionError), (
        "redis.exceptions.AuthenticationError is no longer a subclass of ConnectionError -- "
        "redislib.py's except-AuthenticationError-before-except-ConnectionError ordering was built "
        "on this fact; verify the retry logic still classifies auth failures as permanent."
    )
