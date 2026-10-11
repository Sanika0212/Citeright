"""Benchmark 2: does the cited work back the sentence that cites it?

Two expert-labelled datasets:

1. **SciFact** (Wadden et al., 2020): scientific claims, rewritten from citing sentences,
   against the abstract of the cited paper. Labels: supports, contradicts, not enough
   information. Train 919 pairs, validation 340.
2. **Citation-Integrity** (Sarol et al., 2024): 3,063 real citing sentences from
   biomedical papers against the *full text* of the cited paper, labelled accurate or
   not (contradicts, not substantiated, oversimplified, misquoted, indirect, etiquette).
   Here every label but "accurate" and "contradicts" means the cited work does not
   back the sentence (``not_found``).

For every checker (NLI, MiniCheck, embedding similarity, word overlap) the claim is
scored against the best-matching passages of the cited text (``support.evidence_features``),
a calibrated three-way model is fitted on both training sets, and it is evaluated on
SciFact validation and the Citation-Integrity test split. Reported: accuracy and macro
F1 of the three-way verdict, AUROC and expected calibration error of P(supports), and
the same for the raw checker score with a fixed 0.5 threshold (no calibration).

    citeright bench support --checkers nli minicheck embedding lexical
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from citeright.support import (
    LABELS,
    SUPPORT_FEATURES,
    SupportModel,
    evidence_features,
    make_checker,
)

SCIFACT = "https://huggingface.co/api/datasets/allenai/scifact_entailment/parquet/default/{split}/0.parquet"
CITATION_INTEGRITY = (
    "https://raw.githubusercontent.com/ScienceNLP-Lab/Citation-Integrity/main/Data/"
    "multivers-format/{name}.jsonl"
)
_SCIFACT_LABELS = {"SUPPORT": "supports", "CONTRADICT": "contradicts", "NEI": "not_found"}


def data_dir() -> Path:
    base = Path(os.environ.get("CITERIGHT_CACHE", Path.home() / ".cache" / "citeright"))
    return base / "data"


def _download(url: str, name: str) -> Path:
    path = data_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def scifact(split: str) -> list[dict[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(_download(SCIFACT.format(split=split), f"scifact-{split}.parquet"))
    return [
        {
            "claim": str(row.claim),
            "text": " ".join(str(s) for s in row.abstract),
            "label": _SCIFACT_LABELS[str(row.verdict)],
        }
        for row in frame.itertuples()
    ]


def citation_integrity(split: str) -> list[dict[str, str]]:
    corpus_file = _download(CITATION_INTEGRITY.format(name="corpus"), "citint-corpus.jsonl")
    corpus = {}
    for line in corpus_file.read_text().splitlines():
        doc = json.loads(line)
        corpus[doc["doc_id"]] = " ".join([doc.get("title") or "", *doc.get("abstract", [])])
    name = f"claims-{split}"
    rows = []
    for line in (
        _download(CITATION_INTEGRITY.format(name=name), f"citint-{name}.jsonl")
        .read_text()
        .splitlines()
    ):
        claim = json.loads(line)
        labels = {e["label"] for v in (claim.get("evidence") or {}).values() for e in v}
        label = (
            "contradicts"
            if "CONTRADICT" in labels
            else "supports"
            if labels == {"ACCURATE"}
            else "not_found"
        )
        text = " ".join(corpus.get(d, "") for d in claim.get("cited_doc_ids", []))
        sentence = re.sub(r"<\|[a-z_]+\|>", "", claim["claim"]).strip()
        rows.append({"claim": sentence, "text": text, "label": label})
    return rows


LOADERS = {
    "scifact-train": lambda: scifact("train"),
    "scifact-validation": lambda: scifact("validation"),
    "citint-train": lambda: citation_integrity("train"),
    "citint-test": lambda: citation_integrity("test"),
}
TRAIN = ("scifact-train", "citint-train")
TEST = ("scifact-validation", "citint-test")


def featurize(rows: Sequence[dict[str, str]], checker: Any, top: int) -> np.ndarray:
    out = []
    for n, row in enumerate(rows):
        feats, _ = evidence_features(row["claim"], row["text"], checker, top=top)
        out.append([feats[f] for f in SUPPORT_FEATURES])
        if (n + 1) % 250 == 0:
            print(f"    {n + 1}/{len(rows)}", flush=True)
    return np.asarray(out, dtype=float)


def fit(x: np.ndarray, y: Sequence[str]) -> SupportModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=5000, C=1.0).fit(scaler.transform(x), list(y))
    order = [list(model.classes_).index(label) for label in LABELS]
    # fold the scaling into the weights so inference needs only numpy
    coef = model.coef_[order] / scaler.scale_
    intercept = model.intercept_[order] - (model.coef_[order] * scaler.mean_ / scaler.scale_).sum(1)
    return SupportModel(coef.tolist(), intercept.tolist())


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(model: SupportModel, x: np.ndarray, y: Sequence[str]) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    probs = [model.probabilities(dict(zip(SUPPORT_FEATURES, row, strict=True))) for row in x]
    pred = [max(p, key=lambda k: p[k]) for p in probs]
    p_support = np.array([p["supports"] for p in probs])
    backed = np.array([label == "supports" for label in y], dtype=float)
    raw = x[:, SUPPORT_FEATURES.index("entail_max")]
    raw_pred = ["supports" if r >= 0.5 else "not_found" for r in raw]
    return {
        "n": len(y),
        "accuracy": float(np.mean([a == b for a, b in zip(pred, y, strict=True)])),
        "macro_f1": float(f1_score(list(y), pred, labels=list(LABELS), average="macro")),
        "auroc": _auroc(p_support, backed),
        "ece": _ece(p_support, backed),
        "raw_auroc": _auroc(raw, backed),
        "raw_ece": _ece(np.clip(raw, 0, 1), backed),
        "raw_backed_accuracy": float(
            np.mean([(r == "supports") == bool(b) for r, b in zip(raw_pred, backed, strict=True)])
        ),
        "backed_accuracy": float(
            np.mean([(p == "supports") == bool(b) for p, b in zip(pred, backed, strict=True)])
        ),
        "share_backed": float(backed.mean()),
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="citeright bench support", description=__doc__)
    parser.add_argument(
        "--checkers", nargs="+", default=["nli", "minicheck", "embedding", "lexical"]
    )
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="cap examples per split (0: all)")
    parser.add_argument("--out", default="runs/support")
    parser.add_argument("--models-dir", default="src/citeright/data")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {name: LOADERS[name]() for name in (*TRAIN, *TEST)}
    if args.limit:
        data = {k: v[: args.limit] for k, v in data.items()}
    for name, rows in data.items():
        counts = {label: sum(r["label"] == label for r in rows) for label in LABELS}
        print(f"{name}: {len(rows)} {counts}", flush=True)
    results: dict[str, Any] = {}
    for spec in args.checkers:
        checker = make_checker(spec)
        print(f"== {checker.name}", flush=True)
        feats = {}
        for name, rows in data.items():
            cache = out / f"features-{spec.replace(':', '_').replace('/', '_')}-{name}.npy"
            if cache.exists() and len(np.load(cache)) == len(rows):
                feats[name] = np.load(cache)
            else:
                print(f"  {name}", flush=True)
                feats[name] = featurize(rows, checker, args.top)
                np.save(cache, feats[name])
        x = np.concatenate([feats[n] for n in TRAIN])
        y = [r["label"] for n in TRAIN for r in data[n]]
        model = fit(x, y)
        model.checker = checker.name
        model.fitted_on = f"SciFact train and Citation-Integrity train ({len(y)} claims)"
        model.save(Path(args.models_dir) / f"support_{spec.split(':', 1)[0]}.json")
        results[spec] = {
            name: evaluate(model, feats[name], [r["label"] for r in data[name]]) for name in TEST
        }
        for name in TEST:
            r = results[spec][name]
            print(
                f"  {name:20s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
                f"AUROC {r['auroc']:.3f}  ECE {r['ece']:.3f}",
                flush=True,
            )
    (out / "results.json").write_text(json.dumps(results, indent=1))
    table = markdown(results)
    (out / "results.md").write_text(table)
    print(table)
    return 0


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "| Checker | Dataset | Accuracy (3-way) | Macro F1 | AUROC (backed) | ECE | "
        "Raw score AUROC | Raw score ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for spec, by in results.items():
        for name, r in by.items():
            lines.append(
                f"| {spec} | {name} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | "
                f"{r['auroc']:.3f} | {r['ece']:.3f} | {r['raw_auroc']:.3f} | {r['raw_ece']:.3f} |"
            )
    return "\n".join(lines) + "\n"
