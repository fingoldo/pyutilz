"""Runtime cross-check of a structured field against the free text it was derived from.

A sibling of ``pyutilz.dev.code_audit``, NOT a member of it: every module in that package is an AST
scanner with the signature ``scan_*(root, exclude_dirs=...) -> list[Finding]`` over a source tree,
while this one checks in-memory data RECORDS and cannot be run by the registry or the CLI. Living
inside the scanner package made "everything here is a source scanner" false, so a caller driving the
package by iterating ``get_scanners()`` could never discover this API.

``pyutilz.dev.code_audit`` still re-exports every public name here, and
``pyutilz.dev.code_audit.field_text_agreement`` still resolves (as a shim), so existing imports are
unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field  # aliased: ``FieldTextRule`` has an attribute literally named ``field``
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # import-time-free: importing code_audit here would be circular -- its facade
    # re-exports this module's names, so the package __init__ runs before this module finishes.
    from .code_audit._base import Finding

# --- a structured field that duplicates information also present in free text ------------------------
#
# The defect shape: a record carries an enum field AND the prose the field was derived from, and nothing
# ever compares them. Every such pair is a place where two halves of one claim can disagree indefinitely,
# because each half is validated on its own - the enum against its member list, the prose against the
# source it was copied from - and neither validation can see the other.
#
# Found in a forensic knowledge base, where a `temporal_class` of "postmortem" sat beside an endpoint
# named "vital hanging", i.e. the field said "after death" and the text said "while alive" in the same
# row. Two independent checks passed it: the enum value is legal and the quote is verbatim.
#
# This module is the MECHANISM only. The cue vocabulary is domain knowledge and belongs to the caller:
# what counts as a temporal cue in autopsy prose is not general, and a cue list shipped here would rot
# in a package that cannot measure it. A new field/text pair is therefore a `FieldTextRule` in the
# caller's table, not new code here.
#
# THREE outcomes, not two, and this is the whole point. Folding "the text carries no cue" into "agrees"
# is what let the original defect through: a checker reporting AGREE on every row it could not read
# looks like a clean bill of health and is worth less than no checker. `UNCHECKABLE` is a first-class
# verdict, its count is part of every report, and only CONTRADICT is ever a failure - a cue vocabulary
# is incomplete by construction and a rule that failed on absence would be switched off within a week.

AGREE = "agree"
CONTRADICT = "contradict"
UNCHECKABLE = "uncheckable"

#: The declared value is a "not stated" member (``na``, ``unstated``) while the text names a real one.
#: Reported as a CONTRADICT, under its own kind, because it is a filled text beside an EMPTY field -
#: not agreement. Treating it as agreement is exactly the hole this module was built to close.
KIND_UNFILLED = "unfilled_but_cued"
#: The declared value and the value the text cues cannot both be true of the same record.
KIND_OPPOSED = "declared_opposes_text"


def normalise_text(text: str) -> str:
    """Lowercase, delete INTRA-WORD hyphens and collapse whitespace - applied to cues and text alike.

    Prose hyphenates the same compound inconsistently ("ante-mortem", "antemortem", "post-mortem"), and a
    cue list written one way silently missed the other, which is a checker reading UNCHECKABLE on the
    exact rows it exists for. Deleting the hyphen rather than replacing it with a space is what makes the
    two spellings one token; hyphens between a letter and a digit or at a word edge are left alone.
    """
    lowered = re.sub(r"(?<=[a-z])-(?=[a-z])", "", str(text or "").lower())
    return " ".join(lowered.replace("_", " ").split())


@lru_cache(maxsize=4096)
def _cue_pattern(cue: str) -> re.Pattern[str]:
    """Compiled word-boundary matcher for one normalised cue phrase.

    Cached because a rule's cue list is scanned once per record and a measurement run covers thousands
    of records; compiling per call dominated the cost in the first profile of this module's caller.
    """
    parts = [re.escape(w) for w in normalise_text(cue).split() if w]
    if not parts:
        # A cue that normalises to nothing ("_", "-", whitespace) would compile to a bare
        # double word-boundary, which matches at every word boundary: as a cue it would fire on
        # every record, as an anti-cue it would cancel every cue. An unwritable cue matches
        # nothing instead.
        return re.compile(r"(?!)")
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


@dataclass(frozen=True)
class FieldTextRule:
    """One field whose value is also asserted in free text, plus how to read that text.

    ``name`` names the pair in reports. ``field`` is the record key holding the declared value.
    ``text_fields`` are the record keys whose free text may assert the same thing, in priority order.

    ``cues`` maps a field VALUE to the literal phrases that assert it. Matching is word-boundary and
    hyphen-insensitive; a value with an EMPTY cue list is a table entry whose vocabulary has not been
    written yet, and every record then reads UNCHECKABLE for it rather than agreeing by default. That
    empty state is deliberate and is what makes a new pair cheap to declare and honest to publish.

    ``anti_cues`` maps a field value to phrases that CANCEL its cues, and cancellation is positional: a
    cue counts only where its matched span lies outside every anti-cue span. Without this a vocabulary
    has to choose between missing real cues and firing on homographs - "vital hanging" asserts that the
    person was alive, "injury to vital organs" is anatomy, and "vital" is the same five letters in both.
    A cue vocabulary always needs exceptions, so they are a table column rather than a caller's fork.

    ``neutral_values`` are the members that assert nothing (``na``, ``unstated``). A neutral declared
    value beside a real cue is ``KIND_UNFILLED``, not agreement.

    ``partitions`` groups values that can co-hold. Two values contradict only when both appear in
    partitions and those partitions differ - so a temporal rule can say that ``antemortem`` and
    ``perimortem`` are compatible readings of "vital" while ``postmortem`` is not. ``None`` means every
    value is its own partition, i.e. any mismatch contradicts; that is right for a field like
    ``modality`` where the text licenses exactly one band.

    ``resolver`` overrides ``cues`` entirely: given the whole record it returns ``(value, cue)``, or
    ``("", "")`` when the text says nothing. It exists for the cases where a cue's mere PRESENCE is not
    the question - a negation somewhere in a sentence is not a negation governing the predicate that
    carries the claim - so a caller with a dependency parse can supply scope-aware reading here instead
    of widening the cue list until it lies.
    """

    name: str
    field: str
    text_fields: tuple[str, ...]
    cues: Mapping[str, Sequence[str]] = dc_field(default_factory=dict)
    anti_cues: Mapping[str, Sequence[str]] = dc_field(default_factory=dict)
    neutral_values: frozenset[str] = frozenset()
    partitions: tuple[frozenset[str], ...] | None = None
    resolver: Callable[[Mapping[str, object]], tuple[str, str]] | None = None

    def text_of(self, record: Mapping[str, object]) -> str:
        """The record's free text for this rule: its ``text_fields`` joined, missing keys skipped."""
        return " ".join(str(record.get(k, "") or "") for k in self.text_fields).strip()

    def has_vocabulary(self) -> bool:
        """Whether this rule can read anything at all - a resolver, or at least one non-empty cue list."""
        return self.resolver is not None or any(bool(v) for v in self.cues.values())

    def _partition_of(self, value: str) -> frozenset[str] | None:
        """The partition containing ``value``, or None when the value is not partitioned."""
        for group in self.partitions if self.partitions is not None else ():
            if value in group:
                return group
        return None

    def opposes(self, declared: str, supported: str) -> bool:
        """Whether a declared value and a text-supported value cannot both hold of one record."""
        if declared == supported:
            return False
        if self.partitions is None:
            return True
        pd, ps = self._partition_of(declared), self._partition_of(supported)
        return pd is not None and ps is not None and pd != ps


@dataclass(frozen=True)
class FieldTextVerdict:
    """The outcome of cross-checking one record against one rule.

    ``outcome`` is AGREE / CONTRADICT / UNCHECKABLE. ``kind`` is empty unless ``outcome`` is CONTRADICT,
    where it is ``KIND_UNFILLED`` or ``KIND_OPPOSED``. ``supported`` is the value the text cues (empty
    when nothing was read) and ``cue`` is the phrase that carried it, so a reader can judge the call
    without re-running the checker. ``alternatives`` names the other values the same text cued but
    which lost the tiebreak, so a verdict decided between several readings says so.
    """

    rule: str
    outcome: str
    declared: str
    supported: str
    cue: str
    kind: str = ""
    record_id: str = ""
    alternatives: tuple[str, ...] = ()

    def detail(self) -> str:
        """One-line human reason, safe to put in a report or an exception message."""
        if self.outcome == UNCHECKABLE:
            return f"{self.rule}: no cue in the text; {self.rule} field is {self.declared!r}"
        if self.outcome == AGREE:
            return f"{self.rule}: {self.declared!r} agrees with {self.cue!r}"
        if self.kind == KIND_UNFILLED:
            return f"{self.rule}: field is {self.declared!r} while the text says {self.cue!r} ({self.supported})"
        return f"{self.rule}: field says {self.declared!r} while the text says {self.cue!r} ({self.supported})"

    def as_finding(self, file: str, line: int = 0, severity: str = "P1", snippet: str = "") -> "Finding":
        """Render this verdict as a code_audit ``Finding`` so a caller can report it beside source findings."""
        shown = snippet if snippet else self.cue  # an empty snippet means "not supplied", so fall back to the cue
        from .code_audit._base import Finding

        return Finding(check=f"field_text_{self.rule}", severity=severity, file=file, line=line, snippet=shown, detail=self.detail())


def cues_in_text(rule: FieldTextRule, text: str) -> dict[str, str]:
    """Every field value the text asserts, mapped to the first cue phrase that asserted it.

    Longest cue wins within a value, so a specific phrase is reported rather than a substring of it. A
    cue occurrence swallowed by an anti-cue occurrence does not count, which is how a homograph ("vital
    organs" against "vital hanging") is excluded without weakening the cue itself.
    """
    hits: dict[str, str] = {}
    norm = normalise_text(text)
    for value, phrases in rule.cues.items():
        declared_anti = rule.anti_cues.get(value)
        anti_phrases: Sequence[str] = declared_anti if declared_anti is not None else ()
        blocked = [m.span() for anti in anti_phrases if anti for m in _cue_pattern(anti).finditer(norm)]
        for cue in sorted((p for p in phrases if p), key=len, reverse=True):
            spans = [m.span() for m in _cue_pattern(cue).finditer(norm)]
            if any(not any(lo <= s and e <= hi for lo, hi in blocked) for s, e in spans):
                hits[value] = cue
                break
    return hits


def check_record(rule: FieldTextRule, record: Mapping[str, object], record_id: str = "") -> FieldTextVerdict:
    """Cross-check one record's declared field value against what its own free text asserts.

    UNCHECKABLE whenever the text carries no cue, including when the rule has no vocabulary yet. A
    declared value that is neutral (``na``, ``unstated``) while the text names a real one is a
    CONTRADICT of kind ``KIND_UNFILLED``: the row is not in agreement, it has an unfilled field beside
    a text that fills it.
    """
    declared = str(record.get(rule.field, "") or "")
    alternatives: tuple[str, ...] = ()
    if rule.resolver is not None:
        supported, cue = rule.resolver(record)
    else:
        hits = cues_in_text(rule, rule.text_of(record))
        norm_text = normalise_text(rule.text_of(record))
        # A declared value the text also supports settles the row, even when the text cues several
        # values: prose saying "vital" and "post-mortem" in one sentence supports both readings and the
        # author's choice between them is not a contradiction.
        if declared and declared in hits:
            return FieldTextVerdict(rule.name, AGREE, declared, declared, hits[declared], record_id=record_id)
        # Longest winning cue first, then earliest in the text, and only then alphabetically. Taking
        # the alphabetically-first value made the verdict -- and therefore whether `opposes()` fires
        # at all -- depend on how the values happen to be spelled.
        ordered = sorted(hits.items(), key=lambda kv: (-len(kv[1]), norm_text.find(normalise_text(kv[1])), kv[0]))
        supported, cue = ordered[0] if ordered else ("", "")
        alternatives = tuple(value for value, _cue in ordered[1:])
    if not supported:
        return FieldTextVerdict(rule.name, UNCHECKABLE, declared, "", "", record_id=record_id)
    if declared == supported:
        return FieldTextVerdict(rule.name, AGREE, declared, supported, cue, record_id=record_id, alternatives=alternatives)
    if not declared or declared in rule.neutral_values:
        return FieldTextVerdict(rule.name, CONTRADICT, declared, supported, cue, kind=KIND_UNFILLED, record_id=record_id, alternatives=alternatives)
    if rule.opposes(declared, supported):
        return FieldTextVerdict(rule.name, CONTRADICT, declared, supported, cue, kind=KIND_OPPOSED, record_id=record_id, alternatives=alternatives)
    return FieldTextVerdict(rule.name, AGREE, declared, supported, cue, record_id=record_id, alternatives=alternatives)


@dataclass(frozen=True)
class FieldTextReport:
    """Counts and contradictions for one rule over many records.

    ``coverage`` is the share of records the rule could read at all. It is published beside the verdict
    on purpose: a checker that passes 80% of its input because it found no cue has measured almost
    nothing, and a report of "0 contradictions" without the coverage beside it reads as a clean bill
    of health it has not earned.
    """

    rule: str
    n_records: int
    agree: int
    contradict: int
    uncheckable: int
    contradictions: tuple[FieldTextVerdict, ...]
    has_vocabulary: bool

    @property
    def coverage(self) -> float:
        """Share of records carrying a readable cue - ``(agree + contradict) / n_records``."""
        return (self.agree + self.contradict) / self.n_records if self.n_records else 0.0

    def as_dict(self) -> dict[str, object]:
        """JSON-safe summary: the three counts, the coverage, and the contradiction details."""
        return {
            "rule": self.rule,
            "n_records": self.n_records,
            "agree": self.agree,
            "contradict": self.contradict,
            "uncheckable": self.uncheckable,
            "coverage": round(self.coverage, 4),
            "has_vocabulary": self.has_vocabulary,
            "contradictions": [v.detail() if not v.record_id else f"{v.record_id}: {v.detail()}" for v in self.contradictions],
        }


def check_records(rule: FieldTextRule, records: Iterable[Mapping[str, object]], ids: Iterable[str] | None = None) -> FieldTextReport:
    """Cross-check many records against one rule and summarise agree / contradict / uncheckable.

    ``ids`` optionally labels the records (a pmcid, a row index) so a contradiction can be located
    without re-deriving which record it came from.
    """
    rows = list(records)
    labels = list(ids) if ids is not None else [str(i) for i in range(len(rows))]
    verdicts = [check_record(rule, r, labels[i] if i < len(labels) else str(i)) for i, r in enumerate(rows)]
    return FieldTextReport(
        rule=rule.name,
        n_records=len(rows),
        agree=sum(1 for v in verdicts if v.outcome == AGREE),
        contradict=sum(1 for v in verdicts if v.outcome == CONTRADICT),
        uncheckable=sum(1 for v in verdicts if v.outcome == UNCHECKABLE),
        contradictions=tuple(v for v in verdicts if v.outcome == CONTRADICT),
        has_vocabulary=rule.has_vocabulary(),
    )


def check_all(rules: Iterable[FieldTextRule], records: Iterable[Mapping[str, object]], ids: Iterable[str] | None = None) -> dict[str, FieldTextReport]:
    """Run every rule over the same records; returns one report per rule name.

    Rules with no vocabulary are included rather than skipped, so an unmeasured pair appears in the
    output as 100% uncheckable instead of vanishing from it.
    """
    rows = list(records)
    labels = list(ids) if ids is not None else None
    return {r.name: check_records(r, rows, labels) for r in rules}
