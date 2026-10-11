"""Does the cited work back the sentence that cites it?

1. **Find the passage.** The cited text (abstract, or full text when you provide it) is
   split into sentences and windows of up to three sentences, ranked by BM25 against
   the claim; the top windows go on.
2. **Score it.** A checker scores each window as a premise for the claim:
   P(entails) and P(contradicts). The default is an NLI model; MiniCheck, a
   word-overlap checker and embedding similarity are available for comparison.
3. **Calibrate.** A small multinomial model (``citeright bench support`` fits it on
   SciFact and on Citation-Integrity's expert-labelled biomedical citations) turns the
   strongest window's scores, plus how much of the claim the cited text covers, into
   calibrated probabilities of *supports*, *contradicts* and *not found*.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from citeright.model import Citation, SupportCheck, Verdict
from citeright.text import sentences, tokens

LABELS = ("supports", "contradicts", "not_found")
SUPPORT_FEATURES = (
    "entail_max",
    "entail_top3",
    "contra_max",
    "contra_at_best",
    "coverage",
    "numbers_found",
    "bias_short",
)


class Checker(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        """``(P(entails), P(contradicts))`` of each ``(premise, claim)`` pair."""
        ...


# ---------------------------------------------------------------- passages


def windows(text: str, size: int = 3) -> list[str]:
    sents = sentences(text)
    out = list(sents)
    for width in range(2, size + 1):
        out.extend(" ".join(sents[i : i + width]) for i in range(len(sents) - width + 1))
    return out


def rank(claim: str, passages: Sequence[str], k: int = 6) -> list[str]:
    """BM25 over the candidate passages (k1 1.2, b 0.75)."""
    if len(passages) <= k:
        return list(passages)
    docs = [tokens(p) for p in passages]
    query = set(tokens(claim))
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in query:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])
    return [passages[i] for i in order[:k]]


# ---------------------------------------------------------------- checkers


class LexicalChecker:
    """Share of the claim's content words found in the passage (a baseline; no model)."""

    name = "lexical"

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        out = []
        for premise, claim in pairs:
            words = set(tokens(claim))
            have = set(tokens(premise))
            out.append((len(words & have) / len(words) if words else 0.0, 0.0))
        return out


class _Transformer:
    model_id: str
    device: str | None
    batch_size: int
    _model: Any = None
    _tokenizer: Any = None

    def _load(self, kind: str = "sequence") -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        from citeright.devices import default_device

        self.device = self.device or default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        cls = AutoModelForSequenceClassification if kind == "sequence" else AutoModel
        self._model = cls.from_pretrained(self.model_id).to(self.device).eval()


class NLIChecker(_Transformer):
    """An NLI cross-encoder: entailment and contradiction probabilities."""

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.name = f"nli:{model_id.rsplit('/', 1)[-1]}"
        self.device = device
        self.batch_size = batch_size

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[float, float]]:  # pragma: no cover - needs weights
        import torch

        self._load()
        labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
        ent, con = labels.get("entailment", 0), labels.get("contradiction", 2)
        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self._tokenizer(
                [p for p, _ in batch],
                [c for _, c in batch],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = self._model(**enc).logits.float().softmax(-1).cpu().numpy()
            out.extend((float(p[ent]), float(p[con])) for p in probs)
        return out


class MiniCheckChecker(_Transformer):
    """MiniCheck (RoBERTa-large): P(the passage supports the claim); no contradiction."""

    def __init__(
        self,
        model_id: str = "lytang/MiniCheck-RoBERTa-Large",
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.name = "minicheck"
        self.device = device
        self.batch_size = batch_size

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[float, float]]:  # pragma: no cover - needs weights
        import torch

        self._load()
        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self._tokenizer(
                [p for p, _ in batch],
                [c for _, c in batch],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = self._model(**enc).logits.float().softmax(-1).cpu().numpy()
            out.extend((float(p[1]), 0.0) for p in probs)
        return out


class EmbeddingChecker(_Transformer):
    """Cosine similarity of sentence embeddings (the common shortcut, as a baseline)."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.name = "embedding"
        self.device = device
        self.batch_size = batch_size

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[float, float]]:  # pragma: no cover - needs weights
        import torch

        self._load("embedding")

        def embed(texts: list[str]) -> Any:
            enc = self._tokenizer(
                texts, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1)

        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            a, b = embed([p for p, _ in batch]), embed([c for _, c in batch])
            out.extend((float(s), 0.0) for s in (a * b).sum(-1).cpu().numpy())
        return out


CHECKERS = {
    "nli": NLIChecker,
    "minicheck": MiniCheckChecker,
    "embedding": EmbeddingChecker,
    "lexical": LexicalChecker,
}


def make_checker(spec: str | Checker) -> Checker:
    if not isinstance(spec, str):
        return spec
    name, _, model_id = spec.partition(":")
    cls: Any = CHECKERS[name]
    checker: Checker = cls(model_id) if model_id else cls()
    return checker


# ---------------------------------------------------------------- features


_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def evidence_features(
    claim: str, text: str, checker: Checker, *, top: int = 6
) -> tuple[dict[str, float], str]:
    """Features of the claim against the cited text, and the best passage."""
    passages = rank(claim, windows(text), k=top)
    if not passages:
        return dict.fromkeys(SUPPORT_FEATURES, 0.0), ""
    scores = checker.score([(p, claim) for p in passages])
    entail = np.array([s[0] for s in scores])
    contra = np.array([s[1] for s in scores])
    best = int(np.argmax(entail))
    words = set(tokens(claim))
    numbers = set(_NUMBER.findall(claim))
    return {
        "entail_max": float(entail.max()),
        "entail_top3": float(np.sort(entail)[::-1][:3].mean()),
        "contra_max": float(contra.max()),
        "contra_at_best": float(contra[best]),
        "coverage": len(words & set(tokens(text))) / len(words) if words else 0.0,
        "numbers_found": (
            len({n for n in numbers if n in text}) / len(numbers) if numbers else 1.0
        ),
        "bias_short": float(len(words) <= 6),
    }, passages[best]


@dataclass
class SupportModel:
    """Multinomial logistic model: features -> P(supports), P(contradicts), P(not found)."""

    coef: list[list[float]]  # one row per label in LABELS, one column per feature
    intercept: list[float]
    features: tuple[str, ...] = SUPPORT_FEATURES
    checker: str = ""
    fitted_on: str = ""

    def probabilities(self, feats: dict[str, float]) -> dict[str, float]:
        x = np.array([feats.get(f, 0.0) for f in self.features])
        z = np.asarray(self.coef) @ x + np.asarray(self.intercept)
        p = np.exp(z - z.max())
        p /= p.sum()
        return dict(zip(LABELS, (float(v) for v in p), strict=True))

    @classmethod
    def load(cls, path: str | Path | None = None, checker: str = "nli") -> SupportModel:
        name = f"data/support_{checker.split(':', 1)[0]}.json"
        text = (
            Path(path).read_text()
            if path
            else resources.files("citeright").joinpath(name).read_text()
        )
        data = json.loads(text)
        return cls(
            data["coef"],
            data["intercept"],
            tuple(data["features"]),
            data.get("checker", ""),
            data.get("fitted_on", ""),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "coef": self.coef,
                    "intercept": self.intercept,
                    "features": list(self.features),
                    "checker": self.checker,
                    "fitted_on": self.fitted_on,
                },
                indent=1,
            )
            + "\n"
        )


class SupportChecker:
    """Check citing sentences against the text of the works they cite."""

    def __init__(
        self, checker: str | Checker = "nli", model: SupportModel | None = None, *, top: int = 6
    ) -> None:
        self.checker = make_checker(checker)
        self.top = top
        spec = checker if isinstance(checker, str) else getattr(checker, "name", "nli")
        self.model = model or SupportModel.load(checker=spec)

    def check(self, citation: Citation, key: str, text: str, source: str) -> SupportCheck:
        if not text.strip():
            return SupportCheck(citation, key, Verdict.NO_TEXT)
        feats, evidence = evidence_features(citation.claim, text, self.checker, top=self.top)
        probs = self.model.probabilities(feats)
        label = max(probs, key=lambda k: probs[k])
        verdict = {
            "supports": Verdict.SUPPORTS,
            "contradicts": Verdict.CONTRADICTS,
            "not_found": Verdict.NOT_FOUND,
        }[label]
        return SupportCheck(
            citation,
            key,
            verdict,
            probability=probs["supports"],
            evidence=evidence,
            evidence_score=feats["entail_max"],
            contradiction=probs["contradicts"],
            text_source=source,
        )
