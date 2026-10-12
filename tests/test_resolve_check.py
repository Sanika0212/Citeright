"""Resolution, support checking and whole-draft checks against scripted sources."""

import json
from pathlib import Path

import httpx
import pytest

from citeright.check import check, read_bibliography
from citeright.model import Citation, Record, Reference, Status, Verdict
from citeright.resolve import FEATURES, MatchModel, Resolver, match_features
from citeright.sources import Http, Sources
from citeright.support import LexicalChecker, SupportChecker, SupportModel, rank, windows

DEEP = Record(
    "openalex",
    "W1",
    "Deep learning",
    ["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"],
    2015,
    "10.1038/nature14539",
    venue="Nature",
    abstract="Deep learning allows computational models with multiple layers to learn "
    "representations of data. These methods improved speech recognition and object detection.",
)
RESNET = Record(
    "openalex",
    "W2",
    "Deep residual learning for image recognition",
    ["Kaiming He", "Xiangyu Zhang"],
    2016,
    "10.1109/cvpr.2016.90",
    venue="CVPR",
)
WAKEFIELD = Record(
    "openalex",
    "W3",
    "Ileal-lymphoid-nodular hyperplasia in children",
    ["Andrew Wakefield"],
    1998,
    "10.1016/s0140-6736(97)11096-0",
    retracted=True,
)


class FakeIndex:
    def __init__(self, records: list[Record]) -> None:
        self.records = records
        self.exhausted = False

    def by_doi(self, doi: str) -> Record | None:
        return next((r for r in self.records if r.doi == doi), None)

    def by_id(self, arxiv_id: str) -> Record | None:
        return next((r for r in self.records if r.arxiv == arxiv_id), None)

    def search(self, query: str, limit: int = 5) -> list[Record]:
        words = set(query.lower().split())
        hits = [r for r in self.records if words & set(r.title.lower().split())]
        return hits[:limit]


class FakeCrossref(FakeIndex):
    def by_doi(self, doi: str) -> Record | None:
        if doi == WAKEFIELD.doi:
            notices = [{"type": "retraction", "doi": "10.1016/x", "date": "2010-02-06"}]
            return Record("crossref", doi, WAKEFIELD.title, retracted=True, notices=notices)
        if doi == DEEP.doi:
            notices = [{"type": "correction", "doi": "10.1038/fix", "date": "2016-01-01"}]
            return Record("crossref", doi, DEEP.title, notices=notices)
        return None

    def search(self, query: str, limit: int = 5) -> list[Record]:
        return []


class FakeSources(Sources):
    def __init__(self) -> None:
        records = [DEEP, RESNET, WAKEFIELD]
        self.openalex = FakeIndex(records)  # type: ignore[assignment]
        self.crossref = FakeCrossref(records)  # type: ignore[assignment]
        self.arxiv = FakeIndex([])  # type: ignore[assignment]
        self.europepmc = FakeAbstracts()  # type: ignore[assignment]

    def doi_exists(self, doi: str) -> bool | None:
        return doi in {r.doi for r in (DEEP, RESNET, WAKEFIELD)}


class FakeAbstracts:
    def abstract(self, doi: str) -> str:
        return "Residual networks with 152 layers won ImageNet." if doi == RESNET.doi else ""


MODEL = MatchModel(
    {
        "title_sim": 9.0,
        "first_author": 2.0,
        "author_overlap": 1.0,
        "year_close": 1.0,
        "doi_equal": 1.0,
        "year_far": -1.0,
    },
    bias=-8.0,
    threshold=0.5,
)


@pytest.fixture
def resolver() -> Resolver:
    return Resolver(FakeSources(), MODEL)


def test_match_features_names_and_values() -> None:
    ref = Reference("k", "Deep Learning", ["LeCun, Y."], 2015, doi="10.1038/nature14539")
    feats = match_features(ref, DEEP)
    assert tuple(feats) == FEATURES
    assert feats["title_exact"] == 1.0
    assert feats["first_author"] == 1.0
    assert feats["doi_equal"] == 1.0
    second = match_features(Reference("k", "Deep learning", ["Bengio, Y."]), DEEP)
    assert second["first_author"] == 0.5


def test_verified_with_correction_notice(resolver: Resolver) -> None:
    ref = Reference("k", "Deep learning", ["LeCun, Yann"], 2015, doi="10.1038/nature14539")
    r = resolver.resolve(ref)
    assert r.status is Status.VERIFIED
    assert r.probability > 0.9
    assert r.issues == ["correction published 2016-01-01 (10.1038/fix)"]


def test_mismatches(resolver: Resolver) -> None:
    wrong_year = resolver.resolve(Reference("k", "Deep learning", ["LeCun, Y."], 2011))
    assert wrong_year.status is Status.MISMATCH
    assert wrong_year.issues[0] == "year is 2015, not 2011"
    assert wrong_year.issues[1].startswith("correction published")  # notices are kept
    swapped = resolver.resolve(
        Reference("k", "Deep learning", ["LeCun, Y."], 2015, doi="10.1109/cvpr.2016.90")
    )
    assert swapped.status is Status.MISMATCH
    assert swapped.issues[0].startswith("DOI 10.1109/cvpr.2016.90 belongs to another work")
    assert not any(i.startswith("DOI is") for i in swapped.issues)  # reported once
    fake_doi = resolver.resolve(
        Reference("k", "Deep learning", ["LeCun, Y."], 2015, doi="10.1038/fake123")
    )
    assert fake_doi.status is Status.MISMATCH
    assert fake_doi.issues[0] == "DOI 10.1038/fake123 does not exist"
    author = resolver.resolve(Reference("k", "Deep learning", ["Smith, J."], 2015))
    assert any("not an author" in i for i in author.issues)


def test_not_found_and_retracted(resolver: Resolver) -> None:
    fabricated = resolver.resolve(
        Reference(
            "k",
            "Quantum gradient folding for protein language models",
            ["Zhou, L."],
            2023,
            doi="10.1234/qgf.2023",
        )
    )
    assert fabricated.status is Status.NOT_FOUND
    assert fabricated.issues == ["DOI 10.1234/qgf.2023 does not exist"]
    nothing = resolver.resolve(Reference("k", "Completely unknown words here", ["X, Y"], 2020))
    assert nothing.status is Status.NOT_FOUND
    assert nothing.record is None
    retracted = resolver.resolve(
        Reference("k", "Ileal-lymphoid-nodular hyperplasia in children", ["Wakefield, A."], 1998)
    )
    assert retracted.status is Status.RETRACTED
    assert retracted.issues[0] == "retracted on 2010-02-06 (10.1016/x)"
    assert retracted.to_dict()["status"] == "retracted"


def test_lookup_failure_is_unchecked() -> None:
    class Broken(FakeSources):
        def doi_exists(self, doi: str) -> bool | None:
            raise RuntimeError("service unavailable")

    r = Resolver(Broken(), MODEL).resolve(Reference("k", "Deep learning", doi="10.1/x"))
    assert r.status is Status.UNCHECKED
    assert "service unavailable" in r.issues[0]


def test_match_model_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    MODEL.save(path)
    loaded = MatchModel.load(path)
    assert loaded.weights == MODEL.weights
    assert loaded.probability({"title_sim": 1.0}) == pytest.approx(
        MODEL.probability({"title_sim": 1.0})
    )


SUPPORT = SupportModel(
    coef=[[8.0, 0, -4.0, 0, 0, 0, 0], [0, 0, 8.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
    intercept=[-5.0, -6.0, 0.0],
)


def test_passages_and_ranking() -> None:
    text = "Alpha one. Beta two. Gamma three. Delta four."
    assert len(windows(text)) == 4 + 3 + 2
    ranked = rank("gamma three", windows(text) * 3, k=2)
    assert all("Gamma" in p for p in ranked)


def test_support_checker_verdicts() -> None:
    checker = SupportChecker(LexicalChecker(), SUPPORT)
    cite = Citation("Deep learning improved speech recognition [1].", ["1"])
    good = checker.check(cite, "1", DEEP.abstract, "abstract")
    assert good.verdict is Verdict.SUPPORTS
    assert "speech recognition" in good.evidence
    off = checker.check(
        Citation("Mice prefer cheese to bread [1].", ["1"]), "1", DEEP.abstract, "abstract"
    )
    assert off.verdict is Verdict.NOT_FOUND
    assert checker.check(cite, "1", "  ", "").verdict is Verdict.NO_TEXT
    assert good.to_dict()["citation"]["claim"] == "Deep learning improved speech recognition."


def test_support_model_round_trip(tmp_path: Path) -> None:
    SUPPORT.save(tmp_path / "s.json")
    loaded = SupportModel.load(tmp_path / "s.json")
    assert loaded.probabilities({"entail_max": 1.0}) == SUPPORT.probabilities({"entail_max": 1.0})


DRAFT = r"""
Deep learning improved speech recognition \cite{lecun}. Residual networks
made very deep models trainable \cite{he, ghost}. Vaccines cause autism \cite{wakefield}.
"""
BIB = r"""
@article{lecun, title={Deep learning}, author={LeCun, Yann}, year={2015}, doi={10.1038/nature14539}}
@inproceedings{he, title={Deep residual learning for image recognition},
  author={He, Kaiming}, year={2016}}
@article{wakefield, title={Ileal-lymphoid-nodular hyperplasia in children},
  author={Wakefield, Andrew}, year={1998}}
@article{unused, title={Deep learning}, author={LeCun, Yann}, year={2015}}
"""


def test_check_whole_draft() -> None:
    report = check(
        DRAFT,
        BIB,
        resolver=Resolver(FakeSources(), MODEL),
        support=SupportChecker(LexicalChecker(), SUPPORT),
        workers=2,
    )
    status = {r.reference.key: r.status for r in report.resolutions}
    assert status == {
        "lecun": Status.VERIFIED,
        "he": Status.VERIFIED,
        "wakefield": Status.RETRACTED,
        "unused": Status.VERIFIED,
    }
    assert report.unknown_keys == ["ghost"]
    assert report.uncited == ["unused"]
    verdicts = {c.key: c.verdict for c in report.checks}
    assert verdicts["lecun"] is Verdict.SUPPORTS
    assert verdicts["he"] is not Verdict.NO_TEXT  # abstract found through Europe PMC
    he = next(c for c in report.checks if c.key == "he")
    assert he.text_source == "abstract"
    assert verdicts["wakefield"] is Verdict.NO_TEXT  # no abstract anywhere
    counts = report.counts()
    assert counts["retracted"] == 1
    assert counts["claims_supports"] == 1
    assert report.problems() >= 2
    assert json.loads(json.dumps(report.to_dict()))["unknown_keys"] == ["ghost"]
    no_support = check(DRAFT, BIB, resolver=Resolver(FakeSources(), MODEL), support=False)
    assert no_support.checks == []


def test_read_bibliography_variants() -> None:
    items = r"""Text \cite{a}.
\begin{thebibliography}{9}
\bibitem{a} Y. LeCun, Y. Bengio, and G. Hinton, ``Deep learning,'' \emph{Nature}, 2015.
\bibitem[He et al.(2016)]{b} K. He et al. "Deep residual learning for image recognition."
CVPR, 2016.
\end{thebibliography}"""
    refs = read_bibliography(items, None)
    assert [r.key for r in refs] == ["a", "b"]
    assert refs[1].title == "Deep residual learning for image recognition"
    plain = read_bibliography("x", '[1] A. B, "A title here," 2020.')
    assert plain[0].year == 2020
    given = [Reference("z", "T")]
    assert read_bibliography("x", given) == given
    assert read_bibliography("No references at all.", None) == []


def test_http_cache_retry_and_doi_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "flaky" in str(request.url) and calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        if "handles/10.1/yes" in str(request.url):
            return httpx.Response(200, json={"responseCode": 1})
        if "handles/10.1/no" in str(request.url):
            return httpx.Response(200, json={"responseCode": 100})
        if "missing" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setenv("CITERIGHT_MAILTO", "someone@example.org")
    http = Http(
        cache=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0.0,
    )
    assert http.json("https://x.test/flaky") == {"ok": True}
    assert calls["n"] == 2
    assert http.json("https://x.test/flaky") == {"ok": True}  # cached
    assert calls["n"] == 2
    assert http.json("https://x.test/missing") is None
    assert http.json("https://x.test/keyed", {"q": "a"}) == {"ok": True}
    assert http.json("https://x.test/keyed", {"q": "a", "api_key": "secret"}) == {"ok": True}
    assert calls["n"] == 4  # the key does not change the cache entry
    assert not any("secret" in f.read_text() for f in tmp_path.rglob("*.json"))
    sources = Sources(http)
    assert sources.doi_exists("10.1/yes") is True
    assert sources.doi_exists("10.1/no") is False
    assert Http(cache=False).cache is None


def _report():  # type: ignore[no-untyped-def]
    return check(
        DRAFT,
        BIB,
        resolver=Resolver(FakeSources(), MODEL),
        support=SupportChecker(LexicalChecker(), SUPPORT),
        workers=1,
    )


def test_html_report(tmp_path: Path) -> None:
    from citeright.report import render, write_report

    page = render(_report(), title="Check <draft>")
    assert "<title>Check &lt;draft&gt;</title>" in page
    assert "Retracted" in page
    assert "ghost" in page  # cited but missing
    assert 'href="#ref-lecun"' in page
    assert "<svg" in page
    assert "Problems only" in page
    assert write_report(_report(), tmp_path / "r.html").read_text().startswith("<!doctype html>")


def test_cli_check_and_fail_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import citeright.check as check_module
    from citeright import cli

    monkeypatch.setattr(check_module, "Resolver", lambda: Resolver(FakeSources(), MODEL))
    monkeypatch.setattr(
        "citeright.support.SupportModel.load", lambda *a, **k: SUPPORT, raising=True
    )
    draft, bib = tmp_path / "paper.tex", tmp_path / "refs.bib"
    draft.write_text(DRAFT)
    bib.write_text(BIB)
    base = ["check", str(draft), "--bib", str(bib), "--checker", "lexical"]
    html, out = tmp_path / "r.html", tmp_path / "r.json"
    assert cli.main([*base, "--html", str(html), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "XX wakefield" in text
    assert "cited but not in the bibliography" in text
    assert html.exists()
    assert json.loads(out.read_text())["counts"]["retracted"] == 1
    assert cli.main([*base, "--no-support", "--fail-on", "retracted"]) == 1
    assert cli.main([*base, "--no-support", "--fail-on", "mismatch"]) == 0
    assert cli.main([*base, "--fail-on", "nonsense"]) == 2
    assert cli.main(["bench"]) == 2


def test_cli_resolve(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import citeright.resolve as resolve_module
    from citeright import cli

    original = resolve_module.Resolver
    monkeypatch.setattr(resolve_module, "Resolver", lambda: original(FakeSources(), MODEL))
    assert cli.main(["resolve", "Deep learning", "--author", "LeCun, Y.", "--year", "2015"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


@pytest.mark.parametrize(
    ("claim", "core"),
    [
        ("Smith et al. (2019) showed that aspirin reduces risk.", "Aspirin reduces risk."),
        ("Previous studies have found that sleep helps memory.", "Sleep helps memory."),
        (
            "It has been shown that exercise lowers blood pressure.",
            "Exercise lowers blood pressure.",
        ),
        ("According to the WHO, measles cases rose in 2019.", "Measles cases rose in 2019."),
        ("Deep networks are robust to noise.", "Deep networks are robust to noise."),
        ("We showed that it works.", "We showed that it works."),  # too short to strip
    ],
)
def test_reporting_frames_are_stripped(claim: str, core: str) -> None:
    from citeright.support import core_claim

    assert core_claim(claim) == core
