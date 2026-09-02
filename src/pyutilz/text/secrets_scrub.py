"""Canonical secret-redaction helpers.

Leaf module (stdlib ``re`` only -- no other pyutilz imports), so it can be imported from anywhere
without creating a cycle. Ported from a downstream project's consolidated scrubber, itself the
union of 5+ independently-drifted regex pairs that had accumulated across that codebase (a DSN
scrubber that didn't know about API tokens, a token scrubber that didn't know about DSNs, etc.)
with no shared test suite to catch the drift -- this is meant to be the one place new secret
shapes get added going forward, for every downstream project.

Two entry points, matching two genuinely different use cases:

* :func:`sanitize_dsn` -- mask *only* the password portion of a DSN, preserving the rest
  (host/port/db) for operational debugging. Connection-retry logs want the host visible so
  operators can tell environments apart.
* :func:`redact_secrets` -- the general-purpose "this arbitrary exception text might contain ANY
  secret shape" scrubber for log lines and values about to be persisted. Fully redacts an embedded
  DSN (no host preserved) since callers of this function are handling free-form crash text, not a
  connection string they need to keep partially readable.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# DSN credentials, "scheme://user:pass@host" form (any scheme).  # pragma: allowlist secret
# ---------------------------------------------------------------------------
# The middle password run is greedy and the trailing group anchors to the LAST ``@`` within the
# same non-whitespace run. This is deliberate: it guarantees the credential is ALWAYS fully
# masked, even when the password itself contains ``@`` (e.g. ``user:p@ss@host``). The trade-off is
# cosmetic over-masking when an unrelated ``@`` (e.g. an email address) appears later in the same
# token -- that errs on the safe side (never under-masks / never leaks the password) over a
# non-greedy match which would stop at the first ``@`` and leak the tail of the password.
# The username part is ``*`` not ``+``: ``redis://:password@host`` (empty user) is the standard
# password-only Redis/valkey URL form, and the old ``[^:]+`` skipped it entirely. The password
# run is bounded to non-whitespace so a greedy match cannot span two DSNs separated by ordinary
# text -- that deleted the intervening prose and the first DSN's host.
DSN_PASSWORD_RE = re.compile(r"(://[^:@/\s]*:)[^\s]+(@)")

# libpq also accepts the password as a query-string parameter (``?password=...``); without this,
# DSNs using that form leak in full.
DSN_PASSWORD_QUERY_RE = re.compile(r"([?&]password=)[^&\s]+", re.I)

# Scheme-qualified belt-and-suspenders DSN pattern -- catches a raw DSN embedded mid-sentence in
# free-form exception text, fully redacting it (host included) rather than trying to preserve
# DSN shape.
# ``*`` on the username for the same empty-username reason as DSN_PASSWORD_RE above; redis/amqp
# are the schemes where the password-only DSN form is the common case.
DSN_SCHEME_RE = re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb|rediss?|amqps?)://[^\s:@/]*:[^\s@/]+@\S+")

# ---------------------------------------------------------------------------
# Telegram bot tokens / Bearer / Authorization headers / token & api_key query params.
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN_RE = re.compile(
    # ``Basic`` is included alongside ``Bearer``: without it an ``Authorization: Basic <b64>``
    # header fell through to SECRET_KEY_VALUE_RE, whose value pattern consumed only the word
    # "Basic" and printed the base64-encoded user:pass in full.
    r"/bot[A-Za-z0-9:_-]+/"
    r"|(?<=[?&])token=[^&\s]+"
    r"|(?<=[?&])api[_-]?key=[^&\s]+"
    r"|\b(?:Bearer|Basic)\s+\S+"
    r"|Authorization:\s*(?:Bearer|Basic)\s+\S+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Generic "key=value" / "key: value" secret-shaped fields.
# ---------------------------------------------------------------------------
# Separator: REQUIRES a literal ``:`` or ``=`` (optionally padded with whitespace) between the
# key and the value -- whitespace ALONE as a separator is a real false-positive trap once a broad
# key like "auth" is included: plain English text like "...and oauth flows" or "Unexpected token
# in response" would have a bare space after "auth"/"token" and get spuriously redacted. Requiring
# the punctuation is strictly safer.
#
# ``(?!Bearer\b)`` before the value: applied AFTER ``TELEGRAM_TOKEN_RE`` (see ``redact_secrets``),
# so "Authorization: Bearer <token>" has already been fully redacted to "Authorization: Bearer
# ***" by then -- without this exclusion, this pattern's own "authorization" key alternative would
# re-match that already-redacted text (value = the literal word "Bearer") and mangle it into
# "Authorization=*** ***" (double redaction). "authorization: <token>" with no "Bearer" word is
# unaffected and still redacted normally.
SECRET_KEY_VALUE_RE = re.compile(r"(?i)(password|passwd|api[_-]?key|token|secret|authorization|auth)\s*[:=]\s*(?!(?:Bearer|Basic)\b)\S+")


def sanitize_dsn(text: str) -> str:
    """Mask passwords in DSN strings to prevent credential leakage in logs.

    Preserves the rest of the DSN (host/port/db) so operators can still tell environments apart
    in a connection-retry log line -- only the password itself is masked.

    >>> sanitize_dsn("postgresql://user:s3cret@host/db")  # pragma: allowlist secret
    'postgresql://user:***@host/db'
    >>> sanitize_dsn("OperationalError('connection refused')")
    "OperationalError('connection refused')"
    >>> sanitize_dsn("postgresql://host/db?user=u&password=s3cret&sslmode=require")
    'postgresql://host/db?user=u&password=***&sslmode=require'
    >>> sanitize_dsn("postgresql://user:p@ss@host/db")  # pragma: allowlist secret
    'postgresql://user:***@host/db'
    """
    text = DSN_PASSWORD_RE.sub(r"\1***\2", text)
    text = DSN_PASSWORD_QUERY_RE.sub(r"\1***", text)
    return text


def _telegram_sub(m: "re.Match[str]") -> str:
    """Substitution callback for :data:`TELEGRAM_TOKEN_RE`.

    Preserves a leading ``Bearer``/``Authorization: Bearer`` label so the scrubbed text stays
    readable (``"Bearer ***"`` instead of a bare ``"***"``); the bot-URL-path and
    ``token=``/``api_key=`` query-param alternatives have no useful label to preserve, so they
    collapse to a bare ``"***"``.
    """
    matched = m.group(0)
    lower = matched.lower()
    if lower.startswith("authorization:"):
        scheme = "Basic" if "basic" in lower else "Bearer"
        return f"Authorization: {scheme} ***"
    if lower.startswith("bearer"):
        return "Bearer ***"
    if lower.startswith("basic"):
        return "Basic ***"
    return "***"


def redact_secrets(text: object, limit: Optional[int] = None) -> str:
    """Redact every secret-shaped substring in ``text`` -- the one canonical scrubber for
    arbitrary exception/error text destined for a log line OR a persisted (DB) field. Returns
    ``""`` for falsy input.

    Applies, in order: scheme-qualified DSN (full redaction), generic ``user:pass@`` DSN, DSN
    ``?password=`` query param, Telegram bot-token/Bearer/Authorization forms, then generic
    ``key=value`` secret fields. Order matters: Bearer/Authorization is scrubbed BEFORE the
    generic key=value pass so ``Authorization: Bearer ghp_XYZ`` doesn't get half-eaten by the
    generic pattern (which would consume ``authorization:`` and stop at the first
    whitespace-delimited token, leaving ``Bearer`` and a truncated key visible).

    >>> redact_secrets("postgresql://user:s3cret@host/db")  # pragma: allowlist secret
    '<dsn-redacted>'
    >>> redact_secrets("connect to postgres://admin:s3cret@db.example.com/prod fast")  # pragma: allowlist secret
    'connect to <dsn-redacted> fast'
    >>> redact_secrets("Send Bearer ghp_secret_TOKEN123 to authenticate.")  # pragma: allowlist secret
    'Send Bearer *** to authenticate.'
    >>> redact_secrets("call failed: api_key=sk-super-secret-123 token: deadbeefcafe")  # pragma: allowlist secret
    'call failed: api_key=*** token=***'
    >>> redact_secrets("")
    ''
    >>> redact_secrets(None)
    ''
    """
    if not text:
        return ""
    s = str(text)
    s = DSN_SCHEME_RE.sub("<dsn-redacted>", s)
    s = DSN_PASSWORD_RE.sub(r"\1***\2", s)
    s = DSN_PASSWORD_QUERY_RE.sub(r"\1***", s)
    s = TELEGRAM_TOKEN_RE.sub(_telegram_sub, s)
    s = SECRET_KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}=***", s)
    if limit is not None:
        s = s[:limit]
    return s
