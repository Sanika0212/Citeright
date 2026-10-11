"""Resolve a reference to the scholarly record it means, and say what is wrong with it.

For each reference:

1. **Identifiers first.** A DOI is checked against the DOI handle system (does it exist
   at all?) and looked up in OpenAlex and Crossref; an arXiv id is looked up on arXiv.
2. **Then the title.** Crossref bibliographic search (and OpenAlex title search if that
   fails) finds candidates when there is no identifier, or when it points elsewhere.
3. **Match.** Each candidate is compared field by field (title, authors, year, venue,
   identifiers). A logistic model fitted on real and corrupted references
   (``citeright bench match``) turns the comparison into a calibrated
   P(this record is the work the reference means).
4. **Decide.** ``verified`` when the best match agrees on every field given,
   ``mismatch`` when the work exists but a field is wrong (most often a DOI that belongs
   to another paper, a wrong year or a wrong first author), ``not_found`` when nothing
   matches, ``retracted`` when the matched work carries a retraction notice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from citeright.model import Record, Reference, Resolution, Status
from citeright.sources import Sources
from citeright.text import surname, title_similarity

FEATURES = (
    "title_sim",
    "title_exact",
    "first_author",
    "author_overlap",
    "has_authors",
    "year_close",
    "year_far",
    "has_year",
    "doi_equal",
    "doi_differs",
    "arxiv_equal",
    "venue_sim",
)


def match_features(ref: Reference, rec: Record) -> dict[str, float]:
    """Field-by-field agreement between a reference and a candidate record."""
    title = title_similarity(ref.title or ref.raw, rec.title) if (ref.title or ref.raw) else 0.0
    ref_names = ref.surnames
    rec_names = [surname(a) for a in rec.authors if a.strip()]
    first = 0.0
    if ref_names and rec_names:
        if ref_names[0] == rec_names[0]:
            first = 1.0
        elif ref_names[0] in rec_names:
            first = 0.5
    overlap = (
        len(set(ref_names) & set(rec_names)) / len(set(ref_names)) if ref_names and rec_names else 0
    )
    delta = abs(ref.year - rec.year) if ref.year and rec.year else None
    venue = title_similarity(ref.venue, rec.venue) if ref.venue and rec.venue else 0.0
    return {
        "title_sim": title,
        "title_exact": float(title >= 0.98),
        "first_author": first,
        "author_overlap": float(overlap),
        "has_authors": float(bool(ref_names)),
        "year_close": float(delta is not None and delta <= 1),
        "year_far": float(delta is not None and delta >= 3),
        "has_year": float(ref.year is not None),
        "doi_equal": float(bool(ref.doi and rec.doi and ref.doi == rec.doi)),
        "doi_differs": float(bool(ref.doi and rec.doi and ref.doi != rec.doi)),
        "arxiv_equal": float(bool(ref.arxiv and rec.arxiv and ref.arxiv == rec.arxiv)),
        "venue_sim": venue,
    }


@dataclass
class MatchModel:
    """Logistic model over :data:`FEATURES`, plus the decision threshold it was fitted with."""

    weights: dict[str, float]
    bias: float
    threshold: float = 0.5
    fitted_on: str = ""

    def probability(self, features: dict[str, float]) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))

    @classmethod
    def load(cls, path: str | Path | None = None) -> MatchModel:
        if path is None:
            text = resources.files("citeright").joinpath("data/match_model.json").read_text()
        else:
            text = Path(path).read_text()
        data = json.loads(text)
        return cls(
            data["weights"], data["bias"], data.get("threshold", 0.5), data.get("fitted_on", "")
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "weights": self.weights,
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "fitted_on": self.fitted_on,
                },
                indent=1,
            )
            + "\n"
        )


@dataclass
class _Scored:
    record: Record
    probability: float
    features: dict[str, float] = field(default_factory=dict)


class Resolver:
    def __init__(
        self,
        sources: Sources | None = None,
        model: MatchModel | None = None,
        *,
        search: bool = True,
        candidates: int = 5,
    ) -> None:
        self.sources = sources or Sources()
        self.model = model or MatchModel.load()
        self.search = search
        self.limit = candidates

    def _score(self, ref: Reference, records: Sequence[Record]) -> list[_Scored]:
        out = []
        for rec in records:
            feats = match_features(ref, rec)
            out.append(_Scored(rec, self.model.probability(feats), feats))
        return sorted(out, key=lambda s: -s.probability)

    def _notices(self, rec: Record) -> Record:
        """Attach Crossref's editorial notices (retractions, corrections, concerns)."""
        if rec.source == "crossref" or not rec.doi:
            return rec
        try:
            other = self.sources.crossref.by_doi(rec.doi)
        except RuntimeError:
            return rec
        if other is not None:
            rec.notices = other.notices
            rec.retracted = rec.retracted or other.retracted
            rec.abstract = rec.abstract or other.abstract
        return rec

    def resolve(self, ref: Reference) -> Resolution:
        try:
            return self._resolve(ref)
        except (RuntimeError, OSError, ValueError) as exc:  # network or service failure
            return Resolution(ref, Status.UNCHECKED, issues=[f"lookup failed: {exc}"])

    def _resolve(self, ref: Reference) -> Resolution:
        issues: list[str] = []
        found: list[Record] = []
        doi_record: Record | None = None
        if ref.doi:
            exists = self.sources.doi_exists(ref.doi)
            if exists is False:
                issues.append(f"DOI {ref.doi} does not exist")
            else:
                doi_record = self.sources.openalex.by_doi(ref.doi) or self.sources.crossref.by_doi(
                    ref.doi
                )
                if doi_record is not None:
                    found.append(doi_record)
        if ref.arxiv:
            arxiv_record = self.sources.arxiv.by_id(ref.arxiv)
            if arxiv_record is None:
                issues.append(f"arXiv id {ref.arxiv} does not exist")
            else:
                found.append(arxiv_record)
        scored = self._score(ref, found)
        if self.search and not self._confident(scored) and (ref.title or ref.raw):
            # Crossref first (free); OpenAlex search (daily budget) only if that fails.
            query = _bibliographic(ref) if ref.title else ref.raw
            found.extend(self.sources.crossref.search(query, self.limit))
            scored = self._score(ref, found)
            if not self._confident(scored):
                found.extend(self.sources.openalex.search(ref.title or ref.raw, self.limit))
                scored = self._score(ref, found)
        if not self._confident(scored):
            if doi_record is not None and ref.title:
                issues.append(f"DOI {ref.doi} belongs to another work: {doi_record.title!r}")
            if not issues:
                issues.append("no matching record in OpenAlex, Crossref or arXiv")
            best = scored[0] if scored else None
            return Resolution(
                ref,
                Status.NOT_FOUND,
                best.record if best else None,
                best.probability if best else 0.0,
                issues,
                best.features if best else {},
                len(found),
            )
        best = scored[0]
        record = self._notices(best.record)
        doi_elsewhere = bool(ref.doi) and ref.doi != record.doi and best.record is not doi_record
        if doi_elsewhere and doi_record is not None:
            issues.append(
                f"DOI {ref.doi} belongs to another work ({doi_record.title!r}); "
                f"this work's DOI is {record.doi or 'not registered'}"
            )
        issues.extend(field_issues(ref, record, check_doi=not doi_elsewhere))
        for notice in record.notices:
            if notice["type"] != "retraction":
                label = notice["type"].replace("_", " ")
                issues.append(f"{label} published {notice['date']} ({notice['doi']})")
        if record.retracted:
            status = Status.RETRACTED
            retraction = next((n for n in record.notices if n["type"] == "retraction"), None)
            issues.insert(
                0,
                "retracted"
                + (f" on {retraction['date']} ({retraction['doi']})" if retraction else ""),
            )
        elif any(not _is_notice(i) for i in issues):
            status = Status.MISMATCH
        else:
            status = Status.VERIFIED
        return Resolution(ref, status, record, best.probability, issues, best.features, len(found))

    def _confident(self, scored: Sequence[_Scored]) -> bool:
        return bool(scored) and scored[0].probability >= self.model.threshold

    def resolve_all(self, refs: Sequence[Reference], workers: int = 4) -> list[Resolution]:
        if workers <= 1 or len(refs) <= 1:
            return [self.resolve(r) for r in refs]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.resolve, refs))


def _is_notice(issue: str) -> bool:
    return issue.startswith(("correction", "expression of concern", "erratum", "addendum"))


def _bibliographic(ref: Reference) -> str:
    parts = [ref.title, " ".join(ref.surnames[:3]), str(ref.year or ""), ref.venue]
    return " ".join(p for p in parts if p)


def field_issues(ref: Reference, rec: Record, *, check_doi: bool = True) -> list[str]:
    """Fields the draft got wrong about a work that does exist."""
    issues = []
    if ref.title and title_similarity(ref.title, rec.title) < 0.9:
        issues.append(f"title differs: the record says {rec.title!r}")
    names = [surname(a) for a in rec.authors if a.strip()]
    if ref.surnames and names and ref.surnames[0] not in names:
        issues.append(f"first author {ref.authors[0]!r} is not an author of this work")
    elif ref.surnames and names and ref.surnames[0] != names[0]:
        issues.append(f"first author is {rec.authors[0]!r}, not {ref.authors[0]!r}")
    if ref.year and rec.year and abs(ref.year - rec.year) >= 2:
        issues.append(f"year is {rec.year}, not {ref.year}")
    if check_doi and ref.doi and rec.doi and ref.doi != rec.doi and rec.source != "arxiv":
        issues.append(f"DOI is {rec.doi}, not {ref.doi}")
    if ref.arxiv and rec.arxiv and ref.arxiv != rec.arxiv:
        issues.append(f"arXiv id is {rec.arxiv}, not {ref.arxiv}")
    return issues
