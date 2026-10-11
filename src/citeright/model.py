"""The objects Citeright reads and reports: references, records, citations and findings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    """What resolving a reference found."""

    VERIFIED = "verified"  # a record matches every field that was given
    MISMATCH = (
        "mismatch"  # a record exists, but some fields disagree (or the DOI is another work's)
    )
    NOT_FOUND = "not_found"  # no record matches: possibly fabricated
    RETRACTED = "retracted"  # the matching record has been retracted


class Verdict(StrEnum):
    """Whether the cited work backs the sentence that cites it."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NOT_FOUND = "not_found"  # the cited text does not say this
    NO_TEXT = "no_text"  # no abstract or full text was available to check against


@dataclass
class Reference:
    """A bibliography entry as written in the draft."""

    key: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv: str | None = None
    venue: str = ""
    url: str = ""
    raw: str = ""

    @property
    def surnames(self) -> list[str]:
        from citeright.text import surname

        return [surname(a) for a in self.authors if a.strip()]


@dataclass
class Record:
    """A scholarly record returned by a metadata source."""

    source: str
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv: str | None = None
    venue: str = ""
    abstract: str = ""
    retracted: bool = False
    url: str = ""
    type: str = ""
    # editorial notices attached to the work: {"type": "retraction" | "correction" |
    # "expression_of_concern", "doi": ..., "date": ...}
    notices: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Resolution:
    """The outcome of looking a reference up."""

    reference: Reference
    status: Status
    record: Record | None = None
    probability: float = 0.0  # calibrated P(the record is the work the reference means)
    issues: list[str] = field(default_factory=list)
    checks: dict[str, float] = field(default_factory=dict)
    candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["status"] = self.status.value
        return out


@dataclass
class Citation:
    """A sentence in the draft and the reference keys it cites."""

    sentence: str
    keys: list[str]
    line: int = 0

    @property
    def claim(self) -> str:
        """The sentence with citation markers removed."""
        from citeright.citations import strip_markers

        return strip_markers(self.sentence)


@dataclass
class SupportCheck:
    """Whether one cited work backs one citing sentence, with the passage that decides it."""

    citation: Citation
    key: str
    verdict: Verdict
    probability: float = 0.0  # calibrated P(the cited text supports the claim)
    evidence: str = ""
    evidence_score: float = 0.0
    contradiction: float = 0.0
    text_source: str = ""  # "abstract", "full text" or ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["verdict"] = self.verdict.value
        out["citation"]["claim"] = self.citation.claim
        return out


@dataclass
class Report:
    resolutions: list[Resolution]
    checks: list[SupportCheck]
    unknown_keys: list[str] = field(default_factory=list)  # cited but not in the bibliography
    uncited: list[str] = field(default_factory=list)  # in the bibliography but never cited

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {s.value: 0 for s in Status}
        for r in self.resolutions:
            out[r.status.value] += 1
        for v in Verdict:
            out[f"claims_{v.value}"] = sum(c.verdict is v for c in self.checks)
        return out

    def problems(self) -> int:
        bad_refs = sum(r.status is not Status.VERIFIED for r in self.resolutions)
        bad_claims = sum(c.verdict in (Verdict.CONTRADICTS, Verdict.NOT_FOUND) for c in self.checks)
        return bad_refs + bad_claims + len(self.unknown_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "resolutions": [r.to_dict() for r in self.resolutions],
            "checks": [c.to_dict() for c in self.checks],
            "unknown_keys": self.unknown_keys,
            "uncited": self.uncited,
        }
