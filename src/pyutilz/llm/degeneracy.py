"""Detecting a decoder that has run away into repetition, before its output is trusted or re-bought.

Not hypothetical and not model-specific. Measured on one production corpus: a cheap model emitted 231
structured records for a single source document past a quarantine that missed it, and on a later run
generated more than its own 65,535-token output cap on one document and was truncated. Both are the same
failure - a decoder copying its own context.

**The response is a FLAG, never a ban.** A model that degenerates on one document may be the best
extractor on the next, and banning decides on one observation what a measured rate should decide over
many. So this returns a report; the caller records it, and a model's degeneracy RATE becomes a column to
weigh against its accuracy.

Deterministic and free: no model, no network. It reads only what a repeating decoder leaves behind -
identical records, one span of source text reused across unrelated records, and an implausible count. The
caller supplies what "a record" and "its quoted span" mean, because that is the only domain-bound part.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class DegeneracyThresholds:
    """Where repetition stops being emphasis. Defaults are runaway detectors, not quality bars.

    They are deliberately generous: the point is to catch a decoder that has stopped reading its input, not
    to judge a verbose but honest extraction. A caller whose documents legitimately yield hundreds of
    records should raise `max_records` rather than silence the check.
    """

    max_records: int = 120
    max_duplicate_share: float = 0.30
    max_quote_reuse: int = 8


@dataclass
class DegeneracyReport:
    """The verdict, plus what it was measured on and against - published even when nothing fired.

    A checker that reports only its hits is indistinguishable from one that was never called, so the counts
    and the thresholds travel with the answer.
    """

    flags: list[str] = field(default_factory=list)
    n_records: int = 0
    duplicate_records: int = 0
    max_quote_reuse: int = 0
    thresholds: dict[str, Any] = field(default_factory=dict)

    @property
    def degenerate(self) -> bool:
        """True when any flag fired."""
        return bool(self.flags)

    def as_dict(self) -> dict[str, Any]:
        """A plain dict for a JSON record."""
        return {
            "flags": list(self.flags),
            "degenerate": self.degenerate,
            "n_records": self.n_records,
            "duplicate_records": self.duplicate_records,
            "max_quote_reuse": self.max_quote_reuse,
            "thresholds": dict(self.thresholds),
        }


def degeneracy_report(records: Iterable[tuple[Hashable, str]], thresholds: DegeneracyThresholds | None = None) -> DegeneracyReport:
    """Report repetition in `records`, each a (identity, quoted-span) pair supplied by the caller.

    `identity` is whatever makes two records THE SAME claim in the caller's schema; `quoted-span` is the
    source text the record cites, or "" when it cites none. Whitespace is normalised before comparison so
    that re-wrapped copies of one sentence are recognised as one.
    """
    # Explicit None check rather than `or`: a caller passing a deliberately permissive threshold object
    # must get theirs, and `or` silently substitutes the default for anything falsy.
    th = DegeneracyThresholds() if thresholds is None else thresholds
    pairs = list(records)
    n = len(pairs)
    keys = Counter(key for key, _quote in pairs)
    duplicated = sum(c - 1 for c in keys.values() if c > 1)
    quotes = Counter(_WS.sub(" ", quote.strip().lower()) for _key, quote in pairs if quote and quote.strip())
    most_common = quotes.most_common(1)
    top_reuse = most_common[0][1] if most_common else 0
    flags: list[str] = []
    if n > th.max_records:
        flags.append(f"record_count_implausible:{n}")
    if n and duplicated / n > th.max_duplicate_share:
        flags.append(f"duplicate_record_share:{duplicated}/{n}")
    if top_reuse > th.max_quote_reuse:
        flags.append(f"one_quote_reused:{top_reuse}")
    return DegeneracyReport(
        flags=flags,
        n_records=n,
        duplicate_records=duplicated,
        max_quote_reuse=top_reuse,
        thresholds={"max_records": th.max_records, "max_duplicate_share": th.max_duplicate_share, "max_quote_reuse": th.max_quote_reuse},
    )
