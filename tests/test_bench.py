"""Both benchmark suites end to end on toy data (no network, no model weights)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from citeright.bench import match, run, support
from citeright.model import Record, Reference, Status
from citeright.resolve import FEATURES, MatchModel, Resolver
from citeright.sources import Sources
from tests.test_resolve_check import MODEL

TITLES = [
    "Graph neural networks for molecular property prediction",
    "Attention mechanisms in long document summarization",
    "Calibrated uncertainty for deep ensembles in medicine",
    "Sparse autoencoders reveal interpretable features",
    "Federated learning with differential privacy guarantees",
    "Protein structure prediction from multiple sequence alignments",
    "Causal discovery from observational time series data",
    "Retrieval augmented generation for open domain questions",
]


def records() -> list[Record]:
    return [
        Record(
            "openalex",
            f"W{i}",
            t,
            [f"Alice Smith{i}", f"Bob Jones{i}"],
            2015 + i,
            f"10.1000/{i}",
            venue="Journal",
        )
        for i, t in enumerate(TITLES)
    ]


class Index:
    def __init__(self, recs: list[Record]) -> None:
        self.recs = recs
        self.exhausted = False

    def by_doi(self, doi: str) -> Record | None:
        return next((r for r in self.recs if r.doi == doi), None)

    def search(self, query: str, limit: int = 5) -> list[Record]:
        words = set(query.lower().split())
        scored = sorted(self.recs, key=lambda r: -len(words & set(r.title.lower().split())))
        return [r for r in scored if words & set(r.title.lower().split())][:limit]


class ToySources(Sources):
    def __init__(self, recs: list[Record]) -> None:
        self.http = type("Counter", (), {"requests": 0})()  # type: ignore[assignment]
        self.openalex = Index(recs)  # type: ignore[assignment]
        self.crossref = Index(recs)  # type: ignore[assignment]
        self.arxiv = Index([])  # type: ignore[assignment]

    def doi_exists(self, doi: str) -> bool | None:
        return doi.startswith("10.1000/") and doi.split("/")[1].isdigit()


def test_build_faults_and_expected_status() -> None:
    recs = records()
    retracted = [
        Record(
            "openalex",
            "R1",
            "Retracted study of things and stuff",
            ["Z Q"],
            2010,
            "10.1000/99",
            retracted=True,
        )
    ]
    items = match.build(recs * 6, retracted, seed=1)
    faults = {i["fault"] for i in items}
    assert "retracted" in faults
    assert faults <= set(match.EXPECTED)
    fabricated = [i for i in items if i["fault"] == "fabricated"]
    assert all(i["truth"] is None for i in fabricated)
    ref = match.write_reference(recs[0], "k", random.Random(0))
    assert ref.title == recs[0].title
    assert match._initials("Alice Smith") == "Smith, A."
    assert match._initials("Plato") == "Plato"
    assert match._typo("A short title here", random.Random(3)) != "A short title here"


def test_candidates_baselines_and_fit() -> None:
    recs = records()
    resolver = Resolver(ToySources(recs), MODEL)
    good = Reference("k", TITLES[0], ["Smith0, Alice"], 2015, doi="10.1000/0")
    assert match.candidates(resolver, good)[0].id == "W0"
    assert match.doi_only(resolver, Reference("k", "x", doi="10.1000/zz")) is Status.NOT_FOUND
    assert match.doi_only(resolver, good) is Status.VERIFIED
    assert match.title_search(resolver, good) is Status.VERIFIED
    assert match.title_search(resolver, Reference("k", "Nothing alike")) is Status.NOT_FOUND
    assert match.rules(resolver, good) is Status.VERIFIED
    assert match.rules(resolver, Reference("k", TITLES[1], ["Nobody, N"], 2030)) is Status.MISMATCH
    assert match.rules(resolver, Reference("k", "Unrelated words")) is Status.NOT_FOUND
    truths = {r.id: r.__dict__ for r in recs}
    items = match.build(recs * 4, [], seed=2)
    x, y = match.pair_rows(resolver, items, truths)
    assert x.shape[1] == len(FEATURES)
    assert 0 < y.sum() < len(y)
    model = match.fit_model(x, y)
    assert model.weights["title_sim"] > 0
    assert match.same_work(recs[0], truths["W0"])
    scored = match.score(
        [Status.VERIFIED, Status.MISMATCH], [{"fault": "correct"}, {"fault": "wrong_year"}]
    )
    assert scored["accuracy"] == 1.0
    assert scored["problem_recall"] == 1.0
    table = match.markdown({"results": {"Citeright": scored}})
    assert "| Citeright | 100.0%" in table


def test_match_main_with_cached_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recs = records()
    items = match.build(recs * 10, [], seed=3)
    out = tmp_path / "match"
    out.mkdir()
    (out / "references.json").write_text(
        json.dumps({"items": items, "truths": {r.id: r.__dict__ for r in recs}})
    )
    monkeypatch.setattr(match, "Sources", lambda: ToySources(recs))
    model_path = tmp_path / "model.json"
    assert run("match", ["--out", str(out), "--model-out", str(model_path)]) == 0
    summary = json.loads((out / "results.json").read_text())
    assert set(summary["results"]) == {"Citeright", "rules", "title search", "doi only"}
    assert MatchModel.load(model_path).threshold > 0


class Scripted:
    name = "scripted"

    def score(self, pairs):  # type: ignore[no-untyped-def]
        out = []
        for premise, claim in pairs:
            words = set(claim.lower().split())
            share = len(words & set(premise.lower().split())) / max(len(words), 1)
            out.append((share, 1.0 if "not" in claim.split() else 0.0))
        return out


def toy_rows(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = rng.choice(["supports", "contradicts", "not_found"])
        fact = f"compound {i} reduces inflammation in mice"
        claim = {
            "supports": fact,
            "contradicts": f"compound {i} does not reduce inflammation in mice",
            "not_found": f"gene {i} controls sleep in flies",
        }[label]
        rows.append(
            {"claim": claim, "text": f"We found that {fact}. Other results.", "label": label}
        )
    return rows


def test_support_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaders = {
        name: (lambda s=seed: toy_rows(90, s))
        for seed, name in enumerate([*support.TRAIN, *support.TEST])
    }
    monkeypatch.setattr(support, "LOADERS", loaders)
    monkeypatch.setattr(support, "make_checker", lambda spec: Scripted())
    out = tmp_path / "support"
    argv = ["--checkers", "lexical", "--out", str(out), "--models-dir", str(tmp_path)]
    assert support.main(argv) == 0
    results = json.loads((out / "results.json").read_text())["lexical"]
    for name in support.TEST:
        assert results[name]["accuracy"] > 0.9
        assert results[name]["auroc"] > 0.9
    assert (tmp_path / "support_lexical.json").exists()
    assert support.main([*argv, "--limit", "50"]) == 0  # features recomputed for the cap
    assert np.isnan(support._auroc(np.array([0.1]), np.array([1.0])))


def test_dataset_loaders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    monkeypatch.setenv("CITERIGHT_CACHE", str(tmp_path))
    files = tmp_path / "data"
    files.mkdir()
    pd.DataFrame({"claim": ["c"], "abstract": [["s1", "s2"]], "verdict": ["SUPPORT"]}).to_parquet(
        files / "scifact-train.parquet"
    )
    (files / "citint-corpus.jsonl").write_text(
        json.dumps({"doc_id": 1, "title": "T", "abstract": ["a b"]}) + "\n"
    )
    claims = [
        {
            "claim": "x<|multi_cit|>",
            "evidence": {"1": [{"label": "ACCURATE"}]},
            "cited_doc_ids": [1],
        },
        {"claim": "y", "evidence": {"1": [{"label": "CONTRADICT"}]}, "cited_doc_ids": [1]},
        {"claim": "z", "evidence": {}, "cited_doc_ids": [1]},
    ]
    (files / "citint-claims-test.jsonl").write_text("\n".join(json.dumps(c) for c in claims))
    assert support.scifact("train") == [{"claim": "c", "text": "s1 s2", "label": "supports"}]
    rows = support.citation_integrity("test")
    assert [r["label"] for r in rows] == ["supports", "contradicts", "not_found"]
    assert rows[0]["claim"] == "x"
    assert rows[0]["text"] == "T a b"


def test_auroc_averages_ties() -> None:
    scores = np.array([0.0, 0.0, 0.0, 0.0])
    assert support._auroc(scores, np.array([1, 0, 1, 0])) == 0.5
    assert support._auroc(np.array([0.1, 0.9, 0.5, 0.5]), np.array([0, 1, 1, 0])) == 0.875
