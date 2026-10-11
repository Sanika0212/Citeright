"""Scholarly metadata sources: doi.org, OpenAlex, Crossref and arXiv (all free, no key).

Every request goes through :class:`Http`, which caches responses on disk (so a second
check of the same draft is instant and polite), retries rate limits and server errors
with backoff, and spaces requests out. A contact address for the APIs' "polite pools"
is sent only if you set ``CITERIGHT_MAILTO``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from citeright import __version__
from citeright.model import Record
from citeright.text import normalize_arxiv, normalize_doi

USER_AGENT = f"citeright/{__version__} (+https://github.com/Sanika0212/Citeright)"
_TRANSIENT = {429, 500, 502, 503, 504}


def cache_dir() -> Path:
    return Path(os.environ.get("CITERIGHT_CACHE", Path.home() / ".cache" / "citeright"))


class Http:
    """GET with an on-disk cache, retries and request spacing."""

    def __init__(
        self,
        *,
        cache: Path | str | bool | None = True,
        client: httpx.Client | None = None,
        min_interval: float = 0.12,
        retries: int = 4,
        max_age_days: float = 30.0,
    ) -> None:
        mailto = os.environ.get("CITERIGHT_MAILTO", "")
        agent = f"{USER_AGENT} mailto:{mailto}" if mailto else USER_AGENT
        self.client = client or httpx.Client(
            timeout=30.0, headers={"User-Agent": agent}, follow_redirects=True
        )
        self.cache: Path | None
        if cache is True:
            self.cache = cache_dir() / "http"
        elif cache in (False, None):
            self.cache = None
        else:
            self.cache = Path(str(cache))
        self.min_interval = min_interval
        self.retries = retries
        self.max_age = max_age_days * 86400
        self.requests = 0
        self._last = 0.0

    def _path(self, url: str) -> Path | None:
        if self.cache is None:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.cache / digest[:2] / f"{digest}.json"

    def get(self, url: str, params: dict[str, Any] | None = None) -> tuple[int, str]:
        """``(status, body)``; a 404 is a normal answer ("no such work"), not an error."""
        full = str(httpx.URL(url, params=params or {}))
        path = self._path(full)
        if path is not None and path.exists() and time.time() - path.stat().st_mtime < self.max_age:
            data = json.loads(path.read_text())
            return int(data["status"]), str(data["body"])
        delay = 1.0
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                response = self.client.get(full)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if response.status_code in _TRANSIENT and attempt < self.retries:
                retry_after = response.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else delay)
                delay *= 2
                continue
            status, body = response.status_code, response.text
            if path is not None and status in (200, 404):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"status": status, "body": body}))
            return status, body
        raise RuntimeError(f"unreachable: {full}")  # pragma: no cover

    def json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        status, body = self.get(url, params)
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError(f"{url} returned {status}")
        return json.loads(body)


def doi_exists(http: Http, doi: str) -> bool | None:
    """Ask the DOI handle system directly: True, False, or None if it could not say."""
    try:
        status, body = http.get(f"https://doi.org/api/handles/{quote(doi, safe='/')}")
    except (httpx.HTTPError, RuntimeError):
        return None
    if status == 404:
        return False
    if status != 200:
        return None
    code = json.loads(body).get("responseCode")
    return True if code == 1 else False if code == 100 else None


# --------------------------------------------------------------------- OpenAlex

_OPENALEX_FIELDS = (
    "id,doi,display_name,publication_year,authorships,primary_location,locations,"
    "abstract_inverted_index,is_retracted,type,ids"
)


def _abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, places in inverted.items():
        for place in places:
            slots[place] = word
    return " ".join(slots[i] for i in sorted(slots))


def _openalex_record(work: dict[str, Any]) -> Record:
    arxiv = None
    for location in work.get("locations") or []:
        landing = (location or {}).get("landing_page_url") or ""
        if "arxiv.org" in landing:
            arxiv = normalize_arxiv(landing)
            break
    doi = normalize_doi(work.get("doi"))
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or normalize_arxiv(doi.split("arxiv.", 1)[1])
    source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return Record(
        source="openalex",
        id=str(work.get("id", "")).rsplit("/", 1)[-1],
        title=work.get("display_name") or "",
        authors=[
            (a.get("author") or {}).get("display_name") or "" for a in work.get("authorships") or []
        ],
        year=work.get("publication_year"),
        doi=doi,
        arxiv=arxiv,
        venue=source,
        abstract=_abstract(work.get("abstract_inverted_index")),
        retracted=bool(work.get("is_retracted")),
        url=str(work.get("id", "")),
        type=work.get("type") or "",
    )


class OpenAlex:
    base = "https://api.openalex.org"

    def __init__(self, http: Http) -> None:
        self.http = http

    def by_doi(self, doi: str) -> Record | None:
        work = self.http.json(f"{self.base}/works/doi:{doi}", {"select": _OPENALEX_FIELDS})
        return _openalex_record(work) if work else None

    def by_arxiv(self, arxiv_id: str) -> Record | None:
        return self.by_doi(f"10.48550/arxiv.{arxiv_id}")

    def by_id(self, work_id: str) -> Record | None:
        work = self.http.json(f"{self.base}/works/{work_id}", {"select": _OPENALEX_FIELDS})
        return _openalex_record(work) if work else None

    def search(self, title: str, limit: int = 5) -> list[Record]:
        query = re.sub(r"[,:|!()\[\]{}\"?]", " ", title)
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            return []
        data = self.http.json(
            f"{self.base}/works",
            {"filter": f"title.search:{query}", "per-page": limit, "select": _OPENALEX_FIELDS},
        )
        return [_openalex_record(w) for w in (data or {}).get("results", [])]


# ---------------------------------------------------------------------- Crossref


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _crossref_record(item: dict[str, Any]) -> Record:
    year = None
    for field in ("published", "published-print", "published-online", "issued", "created"):
        parts = (item.get(field) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    authors = []
    for a in item.get("author") or []:
        name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name", "")
        authors.append(name)
    notices = []
    for update in item.get("updated-by") or []:
        kind = str(update.get("type", "")).lower().replace(" ", "_")
        notices.append(
            {
                "type": kind,
                "doi": str(update.get("DOI", "")),
                "date": str((update.get("updated") or {}).get("date-time", ""))[:10],
            }
        )
    titles = item.get("title") or [""]
    venue = (item.get("container-title") or [""])[0] if item.get("container-title") else ""
    return Record(
        source="crossref",
        id=str(item.get("DOI", "")).lower(),
        title=_strip_tags(titles[0]) if titles else "",
        authors=authors,
        year=year,
        doi=normalize_doi(item.get("DOI")),
        venue=venue,
        abstract=_strip_tags(item.get("abstract") or ""),
        retracted=any(n["type"] == "retraction" for n in notices),
        url=str(item.get("URL", "")),
        type=str(item.get("type", "")),
        notices=notices,
    )


class Crossref:
    base = "https://api.crossref.org"

    def __init__(self, http: Http) -> None:
        self.http = http

    def by_doi(self, doi: str) -> Record | None:
        data = self.http.json(f"{self.base}/works/{quote(doi, safe='/()')}")
        return _crossref_record(data["message"]) if data else None

    def search(self, bibliographic: str, limit: int = 5) -> list[Record]:
        if not bibliographic.strip():
            return []
        data = self.http.json(
            f"{self.base}/works", {"query.bibliographic": bibliographic[:500], "rows": limit}
        )
        items = ((data or {}).get("message") or {}).get("items") or []
        return [_crossref_record(i) for i in items]


# ------------------------------------------------------------------------- arXiv

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _arxiv_records(xml: str) -> list[Record]:
    root = ET.fromstring(xml)
    out = []
    for entry in root.findall("a:entry", _ATOM):
        entry_id = entry.findtext("a:id", "", _ATOM)
        title = " ".join(entry.findtext("a:title", "", _ATOM).split())
        if not title or "arxiv.org/api/errors" in entry_id:
            continue
        published = entry.findtext("a:published", "", _ATOM)
        out.append(
            Record(
                source="arxiv",
                id=normalize_arxiv(entry_id) or entry_id,
                title=title,
                authors=[a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)],
                year=int(published[:4]) if published[:4].isdigit() else None,
                doi=normalize_doi(entry.findtext("arxiv:doi", "", _ATOM)),
                arxiv=normalize_arxiv(entry_id),
                venue=entry.findtext("arxiv:journal_ref", "", _ATOM) or "arXiv",
                abstract=" ".join(entry.findtext("a:summary", "", _ATOM).split()),
                url=entry_id,
                type="preprint",
            )
        )
    return out


class ArXiv:
    base = "https://export.arxiv.org/api/query"

    def __init__(self, http: Http) -> None:
        self.http = http

    def by_id(self, arxiv_id: str) -> Record | None:
        status, body = self.http.get(self.base, {"id_list": arxiv_id, "max_results": 1})
        if status != 200:
            return None
        records = _arxiv_records(body)
        return records[0] if records else None

    def search(self, title: str, limit: int = 5) -> list[Record]:
        words = re.sub(r"[^\w\s-]", " ", title).split()
        if not words:
            return []
        query = " AND ".join(f"ti:{w}" for w in words[:12])
        status, body = self.http.get(self.base, {"search_query": query, "max_results": limit})
        return _arxiv_records(body) if status == 200 else []


class Sources:
    """The sources a resolver consults, sharing one HTTP client and cache."""

    def __init__(self, http: Http | None = None) -> None:
        self.http = http or Http()
        self.openalex = OpenAlex(self.http)
        self.crossref = Crossref(self.http)
        self.arxiv = ArXiv(self.http)

    def doi_exists(self, doi: str) -> bool | None:
        return doi_exists(self.http, doi)
