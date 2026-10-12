"""Metadata sources against recorded responses (no network)."""

import json
from pathlib import Path

import httpx
import pytest

from citeright.sources import Http, Sources

OPENALEX_WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.48550/arXiv.1706.03762",
    "display_name": "Attention Is All You Need",
    "publication_year": 2017,
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}, {"author": None}],
    "primary_location": {"source": {"display_name": "arXiv"}},
    "locations": [{"landing_page_url": "https://arxiv.org/abs/1706.03762"}, None],
    "abstract_inverted_index": {"The": [0], "dominant": [1], "models": [2]},
    "is_retracted": False,
    "type": "preprint",
}
CROSSREF_WORK = {
    "DOI": "10.1016/S0140-6736(97)11096-0",
    "title": ["RETRACTED: Ileal-lymphoid-nodular <i>hyperplasia</i>"],
    "author": [{"given": "A J", "family": "Wakefield"}, {"name": "Study Group"}],
    "published": {"date-parts": [[1998, 2, 28]]},
    "container-title": ["The Lancet"],
    "abstract": "<jats:p>Background text.</jats:p>",
    "updated-by": [
        {"DOI": "10.1016/x", "type": "retraction", "updated": {"date-time": "2010-02-06T00:00Z"}},
        {"DOI": "10.1016/y", "type": "correction", "updated": {"date-time": "2004-03-06T00:00Z"}},
    ],
    "URL": "https://doi.org/10.1016/s0140-6736(97)11096-0",
    "type": "journal-article",
}
ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:doi>10.5555/x</arxiv:doi>
  </entry>
</feed>"""


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "api.openalex.org/works/doi:10.48550" in url:
        return httpx.Response(200, json=OPENALEX_WORK)
    if "api.openalex.org/works/W1" in url:
        return httpx.Response(200, json=OPENALEX_WORK)
    if "api.openalex.org/works?" in url and "quota" in url:
        return httpx.Response(429, headers={"retry-after": "3600"})
    if "api.openalex.org/works?" in url and "broken" in url:
        return httpx.Response(500, headers={"retry-after": "0"})
    if "api.openalex.org/works?" in url:
        return httpx.Response(200, json={"results": [OPENALEX_WORK]})
    if "api.crossref.org/works/10.1016" in url:
        return httpx.Response(200, json={"message": CROSSREF_WORK})
    if "api.crossref.org/works?" in url:
        return httpx.Response(200, json={"message": {"items": [CROSSREF_WORK]}})
    if "export.arxiv.org" in url and "9999" in url:
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    if "export.arxiv.org" in url:
        return httpx.Response(200, text=ARXIV_FEED)
    if "ebi.ac.uk/europepmc" in url and "nature14539" in url:
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {"abstractText": "<h4>Abstract</h4>Deep learning allows models to learn."}
                    ]
                }
            },
        )
    if "ebi.ac.uk/europepmc" in url:
        return httpx.Response(200, json={"resultList": {"result": []}})
    if "doi.org/api/handles" in url:
        return httpx.Response(500)
    return httpx.Response(404)


@pytest.fixture
def sources(tmp_path: Path) -> Sources:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Sources(Http(cache=tmp_path, client=client, min_interval=0.0, retries=1))


def test_openalex_records(sources: Sources) -> None:
    rec = sources.openalex.by_arxiv("1706.03762")
    assert rec is not None
    assert rec.title == "Attention Is All You Need"
    assert rec.arxiv == "1706.03762"
    assert rec.doi == "10.48550/arxiv.1706.03762"
    assert rec.abstract == "The dominant models"
    assert rec.authors == ["Ashish Vaswani", ""]
    assert rec.venue == "arXiv"
    assert sources.openalex.by_id("W1") is not None
    assert sources.openalex.search("Attention: is all you need?")[0].id == "W1"
    assert sources.openalex.search("  ") == []
    assert sources.openalex.by_doi("10.9/none") is None


def test_openalex_budget_and_errors(sources: Sources) -> None:
    assert sources.openalex.works("quota") == []
    assert sources.openalex.exhausted
    assert sources.openalex.search("anything") == []  # skipped once the budget is spent
    sources.openalex.exhausted = False
    with pytest.raises(RuntimeError):
        sources.openalex.works("broken")


def test_openalex_api_key(monkeypatch: pytest.MonkeyPatch, sources: Sources) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    from citeright.sources import OpenAlex

    assert OpenAlex(sources.http)._params({"a": 1}) == {"a": 1, "api_key": "k"}


def test_crossref_records_and_notices(sources: Sources) -> None:
    rec = sources.crossref.by_doi("10.1016/S0140-6736(97)11096-0")
    assert rec is not None
    assert rec.title == "RETRACTED: Ileal-lymphoid-nodular hyperplasia"
    assert rec.authors == ["A J Wakefield", "Study Group"]
    assert rec.year == 1998
    assert rec.venue == "The Lancet"
    assert rec.abstract == "Background text."
    assert rec.retracted
    assert [n["type"] for n in rec.notices] == ["retraction", "correction"]
    assert rec.notices[0]["date"] == "2010-02-06"
    assert sources.crossref.search("ileal lymphoid")[0].doi == rec.doi
    assert sources.crossref.search(" ") == []
    assert sources.crossref.by_doi("10.9/none") is None


def test_arxiv_records(sources: Sources) -> None:
    rec = sources.arxiv.by_id("1706.03762")
    assert rec is not None
    assert rec.title == "Attention Is All You Need"
    assert rec.year == 2017
    assert rec.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert rec.arxiv == "1706.03762"
    assert rec.doi == "10.5555/x"
    assert sources.arxiv.by_id("9999.99999") is None
    assert sources.arxiv.search("Attention is all you need")[0].title.startswith("Attention")
    assert sources.arxiv.search("!!") == []


def test_doi_check_unknown_on_server_error(sources: Sources) -> None:
    assert sources.doi_exists("10.1/whatever") is None


def test_http_transport_errors(tmp_path: Path) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    http = Http(
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(fail)),
        min_interval=0.0,
        retries=1,
    )
    with pytest.raises(httpx.ConnectError):
        http.get("https://x.test/")
    ok = Http(
        cache=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
        min_interval=0.0,
        retries=0,
    )
    with pytest.raises(RuntimeError):
        ok.json("https://x.test/a")
    assert json.loads("{}") == {}


def test_europepmc_abstracts(sources: Sources) -> None:
    assert (
        sources.europepmc.abstract("10.1038/nature14539")
        == "Abstract Deep learning allows models to learn."
    )
    assert sources.europepmc.abstract("10.1/none") == ""
