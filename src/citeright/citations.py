"""Find the sentences of a draft that cite something, and the references they cite.

Supported citation styles:

1. LaTeX: ``\\cite{a,b}``, ``\\citep[p.~3]{a}``, ``\\citet``, ``\\parencite``,
   ``\\textcite``, ``\\autocite`` and starred forms.
2. Pandoc Markdown: ``[@a; @b, p. 3]`` and narrative ``@a``.
3. Numbered: ``[1]``, ``[1, 3]``, ``[2-5]``, with a numbered reference list.
4. Author-year in prose: ``(Smith et al., 2020)``, ``Smith and Jones (2019)``.

Reference lists written as plain text (numbered or not) are read by
:func:`parse_reference_list`; fields that cannot be found are left empty, and the raw
string is kept so that a bibliographic search can still resolve it.
"""

from __future__ import annotations

import re

from citeright.model import Citation, Reference
from citeright.text import (
    ARXIV_RE,
    DOI_RE,
    delatex,
    normalize_arxiv,
    normalize_doi,
    sentences,
    surname,
)

LATEX_CITE = re.compile(
    r"\\(?:[Cc]ite[tp]?|[Cc]ite(?:author|year)|parencite|textcite|autocite|footcite|citealp|citealt)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}"
)
PANDOC_BRACKET = re.compile(r"\[([^\[\]]*@[^\[\]]*)\]")
PANDOC_KEY = re.compile(r"(?<![\w.@])-?@([A-Za-z0-9_][\w:.#$%&+?<>~/-]*[\w])")
NUMERIC = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")
AUTHOR_YEAR = re.compile(
    r"(?P<name>[A-Z][A-Za-z'\-]+(?:\s+(?:et al\.?|and|&)\s*(?:[A-Z][A-Za-z'\-]+)?)?)"
    r",?\s*\(?(?P<year>(?:19|20)\d\d)[a-z]?\)?"
)
_MARKERS = (LATEX_CITE, PANDOC_BRACKET, NUMERIC)
REFERENCES_HEADING = re.compile(
    r"^\s*(?:#+\s*|\\section\*?\{)?\s*"
    r"(references|bibliography|works cited|literature cited)\s*\}?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_markers(sentence: str) -> str:
    """The claim a sentence makes, without citation markers or LaTeX."""
    text = sentence
    for pattern in _MARKERS:
        text = pattern.sub("", text)
    text = PANDOC_KEY.sub("", text)
    text = re.sub(r"\(\s*[A-Z][A-Za-z'\-]+(?:\s+et al\.?)?,?\s*(?:19|20)\d\d[a-z]?\s*\)", "", text)
    text = delatex(text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_numbers(spec: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"\s*,\s*", spec):
        bounds = re.split(r"\s*[-–]\s*", part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            lo, hi = int(bounds[0]), int(bounds[1])
            if 0 < hi - lo < 50:
                out.extend(str(n) for n in range(lo, hi + 1))
                continue
        if part.strip().isdigit():
            out.append(part.strip())
    return out


def _strip_latex_noise(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)  # comments
    text = re.sub(  # keep the line count so citations report their real line
        r"\\begin\{(?:equation|align|figure|table)\*?\}.*?\\end\{[^}]*\}",
        lambda m: "\n" * m.group(0).count("\n") or " ",
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"\\(?:section|subsection|paragraph|label|ref)\*?\{[^}]*\}", " ", text)


def body_and_references(text: str) -> tuple[str, str]:
    """Split a draft at its references heading (the last one, if several)."""
    matches = list(REFERENCES_HEADING.finditer(text))
    if not matches:
        bib = re.search(r"\\begin\{thebibliography\}", text)
        return (text[: bib.start()], text[bib.start() :]) if bib else (text, "")
    cut = matches[-1]
    return text[: cut.start()], text[cut.end() :]


def extract_citations(text: str, *, style: str = "auto") -> list[Citation]:
    """Every sentence with at least one citation, with the keys it cites in order."""
    if style == "auto":
        style = (
            "latex"
            if LATEX_CITE.search(text)
            else "pandoc"
            if PANDOC_BRACKET.search(text) or PANDOC_KEY.search(text)
            else "numeric"
            if NUMERIC.search(text)
            else "author-year"
        )
    if style == "latex":
        text = _strip_latex_noise(text)
    body, _ = body_and_references(text)
    out: list[Citation] = []
    for block in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", body):  # paragraphs
        paragraph, cursor = block.group(0), 0
        for sentence in sentences(paragraph):
            words = sentence.split()[:6]
            where = re.compile(r"\s+".join(map(re.escape, words))).search(paragraph, cursor)
            start = where.start() if where else cursor
            cursor = where.end() if where else cursor
            keys = _keys(sentence, style)
            if keys:
                line = body.count("\n", 0, block.start() + start) + 1
                out.append(Citation(sentence=sentence, keys=keys, line=line))
    return out


def _keys(sentence: str, style: str) -> list[str]:
    keys: list[str] = []
    if style == "latex":
        for m in LATEX_CITE.finditer(sentence):
            keys.extend(k.strip() for k in m.group(1).split(",") if k.strip())
    elif style == "pandoc":
        keys.extend(m.group(1) for m in PANDOC_KEY.finditer(sentence))
    elif style == "numeric":
        for m in NUMERIC.finditer(sentence):
            keys.extend(_expand_numbers(m.group(1)))
    else:
        for m in AUTHOR_YEAR.finditer(sentence):
            name = re.split(r"\s+(?:et al|and|&)", m.group("name"))[0]
            keys.append(f"{surname(name)}:{m.group('year')}")
    return list(dict.fromkeys(keys))


def match_author_year(key: str, references: list[Reference]) -> Reference | None:
    """Resolve an author-year key such as ``smith:2020`` against a reference list."""
    name, _, year = key.partition(":")
    hits = [r for r in references if r.year == int(year) and r.surnames[:1] == [name]]
    return hits[0] if len(hits) == 1 else None


_VANCOUVER = re.compile(
    r"^((?:[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)* [A-Z]{1,3}, )*"
    r"[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)* [A-Z]{1,3}(?:,? et al)?)\.\s+"
)
_ENTRY_START = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)\.(?=\s))\s*", re.MULTILINE)


def parse_reference_list(text: str) -> list[Reference]:
    """Read a plain-text reference list (``[1] ...`` or ``1. ...``, else one per paragraph)."""
    _, refs = body_and_references(text)
    refs = refs or text
    starts = list(_ENTRY_START.finditer(refs))
    if starts:
        ends = [m.start() for m in starts[1:]] + [len(refs)]
        chunks = [
            (m.group(1) or m.group(2), refs[m.end() : end])
            for m, end in zip(starts, ends, strict=True)
        ]
    else:
        blocks = [
            b for b in re.split(r"\n\s*\n|\n(?=[A-Z][^\n]{0,80}\(\d{4}\))", refs) if b.strip()
        ]
        chunks = [(str(i + 1), b) for i, b in enumerate(blocks)]
    return [parse_reference_string(key, " ".join(chunk.split())) for key, chunk in chunks]


def parse_reference_string(key: str, raw: str) -> Reference:
    """Best-effort fields from one formatted reference (IEEE, APA, ACL or Vancouver style)."""
    doi = normalize_doi(next(iter(DOI_RE.findall(raw)), None))
    arxiv_match = ARXIV_RE.search(raw)
    arxiv = normalize_arxiv(arxiv_match.group(0)) if arxiv_match else None
    years = re.findall(r"\b(?:19|20)\d\d\b", raw)
    year = int(years[0]) if years else None
    title, authors_part = "", ""
    quoted = re.search(r"[\"“]([^\"”]{8,})[,.]?[\"”]", raw)
    paren_year = re.search(r"\((?:19|20)\d\d[a-z]?\)\.?\s*", raw)
    bare_year = re.search(r"[.,]\s*(?:19|20)\d\d[a-z]?\.\s+", raw)
    if quoted:
        title, authors_part = quoted.group(1), raw[: quoted.start()]
    elif paren_year:  # APA: Authors (2020). Title. Venue.
        title = re.split(r"(?<=[a-z0-9)?])\.\s", raw[paren_year.end() :], maxsplit=1)[0]
        authors_part = raw[: paren_year.start()]
    elif bare_year:  # ACL: Authors. 2020. Title. Venue.
        title = re.split(r"(?<=[a-z0-9)?])\.\s", raw[bare_year.end() :], maxsplit=1)[0]
        authors_part = raw[: bare_year.start()]
    elif vancouver := _VANCOUVER.match(raw):  # Smith J, Lee K. Title. Venue. 2016;
        authors_part = vancouver.group(1)
        title = re.split(r"(?<=[a-z0-9)?])\.\s", raw[vancouver.end() :], maxsplit=1)[0]
    else:
        parts = re.split(r"(?<=[a-z0-9)?])\.\s", raw)
        if len(parts) >= 2:
            authors_part, title = parts[0], parts[1]
    authors = _authors(authors_part)
    return Reference(
        key=key,
        title=delatex(title.strip(" .,")),
        authors=authors,
        year=year,
        doi=doi,
        arxiv=arxiv,
        raw=raw.strip(),
    )


def _authors(part: str) -> list[str]:
    part = re.sub(r"\bet al\.?", "", part).strip(" .,")
    if not part:
        return []
    part = re.sub(r",?\s+(?:and|&)\s+", ", ", part)
    pieces = [p.strip() for p in part.split(",") if p.strip()]
    names: list[str] = []
    i = 0
    while i < len(pieces):
        piece = pieces[i]
        nxt = pieces[i + 1] if i + 1 < len(pieces) else ""
        # "Vaswani, A." -> surname then initials
        if nxt and re.fullmatch(r"(?:[A-Z]\.\s?-?)+(?:[A-Z]\.?)?", nxt):
            names.append(f"{piece}, {nxt}")
            i += 2
            continue
        names.append(piece)
        i += 1
    return [n for n in names if re.search(r"[A-Za-z]{2}", n)]
