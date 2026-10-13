"""Text utilities: LaTeX clean-up, normalisation, names, identifiers, similarity, sentences."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_ACCENTS = {
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    '"': "\u0308",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "u": "\u0306",
    "v": "\u030c",
    "H": "\u030b",
    "c": "\u0327",
    "k": "\u0328",
    "r": "\u030a",
}
_LETTERS = {
    "ss": "ß",
    "o": "ø",
    "O": "Ø",
    "l": "ł",
    "L": "Ł",
    "ae": "æ",
    "AE": "Æ",
    "oe": "œ",
    "OE": "Œ",
    "aa": "å",
    "AA": "Å",
    "i": "ı",
}
_ACCENT_CMD = re.compile(r"\\([`'^\"~=.uvHckr])\s*\{?\s*(\\?[A-Za-z])\s*\}?")
_LETTER_CMD = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i)(?![A-Za-z])\s*(?:\{\})?")
_COMMAND = re.compile(r"\\[A-Za-z]+\*?\s*(?:\[[^\]]*\])?")
_STOP_WORDS = "a an and the of on in for to with by from at as is are via its into using toward"
_STOP = frozenset(_STOP_WORDS.split())
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>{}]+)", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv[:\s/]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)


def delatex(text: str) -> str:
    """Turn LaTeX accents and letters into Unicode and drop other commands and braces."""

    def accent(m: re.Match[str]) -> str:
        base = m.group(2).lstrip("\\")
        base = "i" if base == "i" else base
        return unicodedata.normalize("NFC", base + _ACCENTS[m.group(1)])

    text = _ACCENT_CMD.sub(accent, text)
    text = _LETTER_CMD.sub(lambda m: _LETTERS[m.group(1)], text)
    text = text.replace("\\&", "&").replace("--", "-").replace("~", " ")
    text = re.sub(r"\\(?:textit|textbf|emph|textrm|textsc|mathrm|mbox|text)\s*", "", text)
    text = _COMMAND.sub("", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s+", " ", text).strip()


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize(text: str) -> str:
    """Lower-case ASCII words separated by single spaces (for comparing titles and names)."""
    text = ascii_fold(delatex(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def tokens(text: str, *, stop: bool = True) -> list[str]:
    words = normalize(text).split()
    return [w for w in words if not (stop and w in _STOP)]


_PARTICLES = {
    "al", "da", "das", "de", "del", "della", "den", "der", "di", "do", "dos", "du", "el",
    "la", "le", "ten", "ter", "van", "von",
}  # fmt: skip


def surname(name: str) -> str:
    """``"Vaswani, Ashish"``, ``"Ashish Vaswani"``, ``"A. Vaswani"`` and ``"Vaswani A"`` all
    give ``vaswani``. Particles and hyphens are dropped, since sources disagree on them:
    ``"de Oliveira"`` and ``"Oliveira"``, ``"El-Hassouni"`` and ``"El Hassouni"`` agree."""
    name = delatex(name).strip()
    if "," in name:
        last = name.split(",", 1)[0]
    else:
        parts = name.split()
        if not parts:
            return ""
        if len(parts) >= 2 and re.fullmatch(r"[A-Z]{1,3}", parts[-1]):
            last = " ".join(parts[:-1])  # Vancouver style: "He K", "Smith JA"
        else:
            last = parts[-1]
            for i, part in enumerate(parts[:-1]):  # "Ludwig van Beethoven"
                if part.lower() in _PARTICLES:
                    last = " ".join(parts[i:])
                    break
    words = normalize(last).split()
    return " ".join([w for w in words if w not in _PARTICLES] or words[-1:])


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = doi.rstrip(".,;)")
    return doi.lower() if doi.startswith("10.") else None


def normalize_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    match = ARXIV_RE.search(value) or ARXIV_RE.search(f"arxiv:{value.strip()}")
    return match.group(1).lower() if match else None


_RETRACTED = re.compile(r"^(?:retracted(?: article)?|retraction|withdrawn)\s+")


def title_similarity(a: str, b: str) -> float:
    """Similarity of two titles in [0, 1], robust to case, accents, punctuation and LaTeX.

    The larger of a character-level ratio (catches typos) and a token-set overlap
    (catches reordering and dropped subtitles), with containment counted when the
    shorter title is a prefix of the longer (a subtitle left out).
    """
    na, nb = _RETRACTED.sub("", normalize(a)), _RETRACTED.sub("", normalize(b))
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()) - _STOP, set(nb.split()) - _STOP
    if not ta or not tb:
        return ratio
    overlap = len(ta & tb) / len(ta | tb)
    short, long_ = sorted((na, nb), key=len)
    prefix = 0.95 if long_.startswith(short) and len(short.split()) >= 3 else 0.0
    return max(ratio, overlap, prefix)


_ABBREV = re.compile(
    r"\b(?:et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Sec|Ref|Refs|No|Vol|al|approx|resp|Dr|Mr|Ms|St)"
    r"\.$",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking at "et al.", "e.g.", "Fig." or decimals."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9\\])", text):
        candidate = text[start : m.end()].strip()
        if _ABBREV.search(text[start : m.start() + 1]) or re.search(r"\b[A-Z]\.$", candidate):
            continue  # abbreviation or an initial: not a sentence end
        out.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
