"""Benchmark 1: is each reference real, correct and not retracted?

**Data.** Real references are the reference lists of a random sample of recent articles
in OpenAlex, across all fields. Each reference is written the way an author would
(title, authors, year, venue, and a DOI half the time) and then given at most one fault:

1. ``correct``: unchanged (expected: verified);
2. ``typo``: a letter swapped or a word dropped in the title (expected: verified);
3. ``wrong_year``: the year moved by 2 to 4 (expected: mismatch);
4. ``wrong_author``: the first author replaced by an author of another paper (mismatch);
5. ``doi_swap``: the DOI of another paper in the sample (mismatch);
6. ``fake_doi``: a DOI with the right prefix and an invented suffix (mismatch);
7. ``fabricated``: a title spliced from two real titles, other authors, an invented DOI
   half the time (expected: not found);
8. ``retracted``: a retracted work from OpenAlex, written correctly (expected: retracted).

**Model.** For every reference all candidates are gathered (DOI and arXiv lookups,
Crossref bibliographic search, OpenAlex title search) and labelled by whether they are
the true record. A logistic model over the field comparisons is fitted on the training
split; its threshold is chosen on the development split; everything is reported on the
test split, and the model is written to ``src/citeright/data/match_model.json``.

**Baselines.** ``doi only`` (flag a reference only when its DOI does not resolve, as
link checkers do), ``title search`` (verified when the best search hit has a near-identical
title) and ``rules`` (title, first author and year within hand-set tolerances).

    citeright bench match --seeds 60 --per-seed 12 --retracted 60
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from citeright.model import Record, Reference, Status
from citeright.resolve import FEATURES, MatchModel, Resolver, field_issues, match_features
from citeright.sources import Sources
from citeright.text import surname, title_similarity

EXPECTED = {
    "correct": Status.VERIFIED,
    "typo": Status.VERIFIED,
    "wrong_year": Status.MISMATCH,
    "wrong_author": Status.MISMATCH,
    "doi_swap": Status.MISMATCH,
    "fake_doi": Status.MISMATCH,
    "fabricated": Status.NOT_FOUND,
    "retracted": Status.RETRACTED,
}
FAULTS = ("correct", "typo", "wrong_year", "wrong_author", "doi_swap", "fake_doi", "fabricated")
WEIGHTS = (3, 1, 1, 1, 1, 1, 2)


# ------------------------------------------------------------------ data


def sample_records(sources: Sources, seeds: int, per_seed: int, seed: int) -> list[Record]:
    """Works cited by a random sample of recent, cited articles."""
    oa = sources.openalex
    papers = oa.http.json(
        f"{oa.base}/works",
        oa._params({
            "filter": "publication_year:2021-2024,type:article,has_references:true,"
                      "cited_by_count:>5",
            "sample": seeds, "seed": seed, "per-page": seeds, "select": "id,referenced_works",
        }),
    )  # fmt: skip
    rng = random.Random(seed)
    wanted: list[str] = []
    for paper in (papers or {}).get("results", []):
        refs = [r.rsplit("/", 1)[-1] for r in paper.get("referenced_works") or []]
        rng.shuffle(refs)
        wanted.extend(refs[:per_seed])
    wanted = list(dict.fromkeys(wanted))
    records: list[Record] = []
    for i in range(0, len(wanted), 50):
        batch = "|".join(wanted[i : i + 50])
        records.extend(oa.works(f"openalex_id:{batch}", limit=50))
    usable = [
        r for r in records
        if r.title and len(r.title.split()) >= 4 and r.authors and r.year and not r.retracted
    ]  # fmt: skip
    return usable


def sample_retracted(sources: Sources, n: int, seed: int) -> list[Record]:
    if n <= 0:
        return []
    found = sources.openalex.works(
        "is_retracted:true,has_doi:true,type:article,publication_year:2000-2023",
        limit=min(n, 100), sample=n, seed=seed,
    )  # fmt: skip
    return [r for r in found if r.title and r.authors and r.year]


def _initials(name: str) -> str:
    parts = name.replace(",", " ").split()
    if len(parts) < 2:
        return name
    return f"{parts[-1]}, {' '.join(p[0] + '.' for p in parts[:-1])}"


def write_reference(rec: Record, key: str, rng: random.Random) -> Reference:
    """How an author might cite ``rec``: full names or initials, a DOI half the time."""
    authors = rec.authors[: rng.choice([3, 5, 10])]
    if rng.random() < 0.5:
        authors = [_initials(a) for a in authors]
    title = rec.title.removeprefix("RETRACTED: ").removeprefix("RETRACTED ARTICLE: ")
    return Reference(
        key=key,
        title=title,
        authors=authors,
        year=rec.year,
        doi=rec.doi if rng.random() < 0.5 else None,
        venue=rec.venue,
    )


def _typo(title: str, rng: random.Random) -> str:
    words = title.split()
    if len(words) > 6 and rng.random() < 0.5:
        del words[rng.randrange(1, len(words) - 1)]
        return " ".join(words)
    i = rng.randrange(1, max(2, len(title) - 2))
    return title[:i] + title[i + 1] + title[i] + title[i + 2 :]


def build(
    records: Sequence[Record], retracted: Sequence[Record], seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    items: list[dict[str, Any]] = []
    for i, rec in enumerate(pool):
        fault = rng.choices(FAULTS, WEIGHTS)[0]
        ref = write_reference(rec, f"r{i}", rng)
        other = pool[(i + 1 + rng.randrange(len(pool) - 1)) % len(pool)]
        truth: str | None = rec.id
        if fault == "typo":
            ref.title = _typo(ref.title, rng)
        elif fault == "wrong_year":
            ref.year = (rec.year or 2020) + rng.choice([-4, -3, -2, 2, 3, 4])
        elif fault == "wrong_author":
            ref.authors = [other.authors[0], *ref.authors[1:]]
            if surname(other.authors[0]) in {surname(a) for a in rec.authors}:
                fault = "correct"
        elif fault == "doi_swap":
            if not other.doi or other.doi == rec.doi:
                fault = "correct"
            else:
                ref.doi = other.doi
        elif fault == "fake_doi":
            prefix = (rec.doi or "10.1016/x").split("/", 1)[0]
            suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
            ref.doi = f"{prefix}/{suffix}"
        elif fault == "fabricated":
            donor = pool[(i + 7) % len(pool)]
            a, b = rec.title.split(), donor.title.split()
            ref.title = " ".join(a[: max(2, len(a) // 2)] + b[len(b) // 2 :])
            ref.authors = donor.authors[:3]
            ref.year = rng.randint(2012, 2024)
            ref.venue = donor.venue
            ref.doi = None
            if rng.random() < 0.5:
                suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))
                ref.doi = f"{(rec.doi or '10.1016/x').split('/', 1)[0]}/{suffix}"
            truth = None
        items.append({"fault": fault, "truth": truth, "reference": asdict(ref)})
    for j, rec in enumerate(retracted):
        ref = write_reference(rec, f"x{j}", rng)
        items.append({"fault": "retracted", "truth": rec.id, "reference": asdict(ref)})
    rng.shuffle(items)
    return items


# ------------------------------------------------------------- candidates


def candidates(resolver: Resolver, ref: Reference) -> list[Record]:
    """Every candidate the resolver could consider, without its early stops."""
    s = resolver.sources
    out: list[Record] = []
    if ref.doi and s.doi_exists(ref.doi) is not False:
        rec = s.openalex.by_doi(ref.doi) or s.crossref.by_doi(ref.doi)
        if rec:
            out.append(rec)
    if ref.title:
        found = s.crossref.search(" ".join([ref.title, *ref.surnames[:2]]), 5)
        out.extend(found)
        if not any(title_similarity(ref.title, r.title) >= 0.9 for r in found):
            out.extend(s.openalex.search(ref.title, 5))  # costs budget: only when needed
    return out


def same_work(rec: Record, truth: dict[str, Any]) -> bool:
    if truth.get("doi") and rec.doi:
        return bool(rec.doi == truth["doi"])
    return rec.id == truth["id"] or title_similarity(rec.title, truth["title"]) > 0.97


def pair_rows(
    resolver: Resolver, items: Sequence[dict[str, Any]], truths: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for item in items:
        ref = Reference(**item["reference"])
        truth = truths.get(item["truth"] or "")
        for rec in candidates(resolver, ref):
            feats = match_features(ref, rec)
            xs.append([feats[f] for f in FEATURES])
            ys.append(int(truth is not None and same_work(rec, truth)))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def fit_model(x: np.ndarray, y: np.ndarray) -> MatchModel:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=5000).fit(x, y)
    return MatchModel(
        dict(zip(FEATURES, (float(w) for w in model.coef_[0]), strict=True)),
        float(model.intercept_[0]),
    )


# --------------------------------------------------------------- baselines


def doi_only(resolver: Resolver, ref: Reference) -> Status:
    if ref.doi and resolver.sources.doi_exists(ref.doi) is False:
        return Status.NOT_FOUND
    return Status.VERIFIED


def title_search(resolver: Resolver, ref: Reference) -> Status:
    hits = resolver.sources.crossref.search(ref.title, 5)
    best = max((title_similarity(ref.title, h.title) for h in hits), default=0.0)
    if best < 0.9:
        hits = resolver.sources.openalex.search(ref.title, 5)
        best = max([best, *(title_similarity(ref.title, h.title) for h in hits)])
    return Status.VERIFIED if best >= 0.9 else Status.NOT_FOUND


def rules(resolver: Resolver, ref: Reference) -> Status:
    """Hand-set tolerances: the usual way reference checkers are built."""
    best = None
    for rec in candidates(resolver, ref):
        if title_similarity(ref.title, rec.title) >= 0.9:
            best = rec
            break
    if best is None:
        return Status.NOT_FOUND
    if best.retracted:
        return Status.RETRACTED
    return Status.MISMATCH if field_issues(ref, best) else Status.VERIFIED


# ------------------------------------------------------------------- run


def score(statuses: Sequence[Status], items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = [EXPECTED[i["fault"]] for i in items]
    correct = [s is e for s, e in zip(statuses, expected, strict=True)]
    flagged = [s is not Status.VERIFIED for s in statuses]
    faulty = [e is not Status.VERIFIED for e in expected]
    tp = sum(f and t for f, t in zip(flagged, faulty, strict=True))
    by_fault: dict[str, list[bool]] = defaultdict(list)
    for item, ok in zip(items, correct, strict=True):
        by_fault[item["fault"]].append(ok)
    return {
        "accuracy": float(np.mean(correct)),
        "problem_precision": tp / max(sum(flagged), 1),
        "problem_recall": tp / max(sum(faulty), 1),
        "by_fault": {k: float(np.mean(v)) for k, v in sorted(by_fault.items())},
        "n": len(items),
        "unchecked": sum(s is Status.UNCHECKED for s in statuses),  # lookups that failed
    }


def _budget_spent(sources: Sources) -> bool:
    return sources.openalex.exhausted


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="citeright bench match", description=__doc__)
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--per-seed", type=int, default=12)
    parser.add_argument("--retracted", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/match")
    parser.add_argument("--model-out", default="src/citeright/data/match_model.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = Sources()
    data_file = out / "references.json"
    if data_file.exists():
        data = json.loads(data_file.read_text())
    else:
        records = sample_records(sources, args.seeds, args.per_seed, args.seed)
        retracted = sample_retracted(sources, args.retracted, args.seed)
        truths = {r.id: asdict(r) for r in [*records, *retracted]}
        data = {"items": build(records, retracted, args.seed), "truths": truths}
        if sources.openalex.exhausted:  # never keep a short sample
            raise SystemExit("OpenAlex's daily budget ran out while sampling; try again later")
        data_file.write_text(json.dumps(data, indent=1))
    items, truths = data["items"], data["truths"]
    print(f"{len(items)} references: {dict(Counter(i['fault'] for i in items))}", flush=True)
    n = len(items)
    train, dev, test = items[: n * 3 // 5], items[n * 3 // 5 : n * 4 // 5], items[n * 4 // 5 :]
    placeholder = MatchModel(dict.fromkeys(FEATURES, 0.0), 0.0)
    gather = Resolver(sources, placeholder)
    x, y = pair_rows(gather, train, truths)
    print(f"training pairs: {len(y)} ({int(y.sum())} true matches)", flush=True)
    model = fit_model(x, y)
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.2, 0.96, 0.05):
        model.threshold = float(t)
        resolver = Resolver(sources, model)
        acc = score([resolver.resolve(Reference(**i["reference"])).status for i in dev], dev)[
            "accuracy"
        ]
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    model.threshold = best_t
    model.fitted_on = (
        f"{len(train)} references ({len(y)} candidate pairs) from OpenAlex reference lists; "
        f"threshold {best_t:.2f} chosen on {len(dev)} development references"
    )
    model.save(args.model_out)
    print(f"threshold {best_t:.2f} (dev accuracy {best_acc:.3f})", flush=True)
    resolver = Resolver(sources, model)
    results: dict[str, Any] = {}
    methods: dict[str, Callable[[Reference], Status]] = {
        "Citeright": lambda r: resolver.resolve(r).status,
        "rules": lambda r: rules(resolver, r),
        "title search": lambda r: title_search(resolver, r),
        "doi only": lambda r: doi_only(resolver, r),
    }
    for name, method in methods.items():
        statuses = [method(Reference(**i["reference"])) for i in test]
        results[name] = score(statuses, test)
        print(f"  {name:12s} accuracy {results[name]['accuracy']:.3f}", flush=True)
    if any(r["unchecked"] for r in results.values()):
        print("warning: some lookups failed (network); rerun to fill them from the cache")
    exhausted = _budget_spent(sources)
    if exhausted:
        print("warning: OpenAlex's search budget ran out; some searches were skipped")
    summary = {
        "openalex_exhausted": exhausted,
        "references": n,
        "test": len(test),
        "faults": dict(Counter(i["fault"] for i in test)),
        "model": {"weights": model.weights, "bias": model.bias, "threshold": model.threshold},
        "results": results,
        "requests": sources.http.requests,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    table = markdown(summary)
    (out / "results.md").write_text(table)
    print(table)
    return 0


def markdown(summary: dict[str, Any]) -> str:
    faults = list(EXPECTED)
    head = "| Method | Accuracy | Problem precision | Problem recall | " + " | ".join(faults) + " |"
    lines = [head, "|" + "---|" * (4 + len(faults))]
    for name, r in summary["results"].items():
        cells = [f"{r['by_fault'].get(f, float('nan')):.0%}" for f in faults]
        lines.append(
            f"| {name} | {r['accuracy']:.1%} | {r['problem_precision']:.1%} | "
            f"{r['problem_recall']:.1%} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"
