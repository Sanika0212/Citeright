"""A small, forgiving BibTeX reader (no dependencies).

Handles braces nested to any depth, quoted values, ``#`` concatenation, ``@string``
macros, month names, ``@comment`` and ``@preamble`` blocks, and entries that are cut off.
"""

from __future__ import annotations

import re

from citeright.model import Reference
from citeright.text import delatex, normalize_arxiv, normalize_doi

_MONTHS = {m: m for m in "jan feb mar apr may jun jul aug sep oct nov dec".split()}
_SKIP = {"comment", "preamble"}


class _Scanner:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def skip_space(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def braced(self) -> str:
        """Read ``{...}`` (the scanner is on the opening brace) and return the inside."""
        depth, start = 0, self.i
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch == "\\":
                self.i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return self.text[start + 1 : self.i - 1]
            self.i += 1
        return self.text[start + 1 :]

    def quoted(self) -> str:
        self.i += 1
        start, depth = self.i, 0
        while self.i < len(self.text):
            ch = self.text[self.i]
            if ch == "\\":
                self.i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == '"' and depth == 0:
                self.i += 1
                return self.text[start : self.i - 1]
            self.i += 1
        return self.text[start:]

    def word(self) -> str:
        m = re.compile(r"[^\s,=#{}\"()]+").match(self.text, self.i)
        if not m:
            return ""
        self.i = m.end()
        return m.group(0)

    def value(self, macros: dict[str, str]) -> str:
        parts = []
        while True:
            self.skip_space()
            ch = self.peek()
            if ch == "{":
                parts.append(self.braced())
            elif ch == '"':
                parts.append(self.quoted())
            else:
                w = self.word()
                parts.append(macros.get(w.lower(), w))
            self.skip_space()
            if self.peek() == "#":
                self.i += 1
                continue
            return "".join(parts)


def parse_entries(text: str) -> list[tuple[str, str, dict[str, str]]]:
    """``(entry type, key, fields)`` for every entry, with raw (still LaTeX) values."""
    scanner = _Scanner(text)
    macros: dict[str, str] = dict(_MONTHS)
    entries = []
    while True:
        at = text.find("@", scanner.i)
        if at < 0:
            break
        scanner.i = at + 1
        kind = scanner.word().lower()
        scanner.skip_space()
        if scanner.peek() not in "{(" or not kind:
            continue
        close = "}" if scanner.peek() == "{" else ")"
        if kind in _SKIP:
            if close == "}":
                scanner.braced()
            continue
        scanner.i += 1
        if kind == "string":
            scanner.skip_space()
            name = scanner.word().lower()
            scanner.skip_space()
            if scanner.peek() == "=":
                scanner.i += 1
                macros[name] = scanner.value(macros)
            continue
        scanner.skip_space()
        key = scanner.word()
        fields: dict[str, str] = {}
        while scanner.i < len(text):
            scanner.skip_space()
            if scanner.peek() == ",":
                scanner.i += 1
                continue
            if scanner.peek() == close or scanner.peek() == "":
                scanner.i += 1
                break
            if scanner.peek() == "@":  # the entry was cut off; start the next one
                break
            name = scanner.word().lower()
            if not name:
                scanner.i += 1
                continue
            scanner.skip_space()
            if scanner.peek() != "=":
                continue
            scanner.i += 1
            fields[name] = scanner.value(macros).strip()
        entries.append((kind, key, fields))
    return entries


def split_authors(value: str) -> list[str]:
    """BibTeX ``and``-separated names, without ``and others``."""
    names = re.split(r"\s+and\s+", value.strip(), flags=re.IGNORECASE)
    return [delatex(n).strip() for n in names if n.strip() and n.strip().lower() != "others"]


def _year(fields: dict[str, str]) -> int | None:
    for name in ("year", "date"):
        m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", fields.get(name, ""))
        if m:
            return int(m.group(1))
    return None


def to_reference(kind: str, key: str, fields: dict[str, str], raw: str = "") -> Reference:
    venue = next(
        (fields[f] for f in ("journal", "journaltitle", "booktitle", "publisher") if f in fields),
        "",
    )
    eprint = fields.get("eprint", "")
    arxiv = None
    if fields.get("archiveprefix", "").lower() == "arxiv" or "arxiv" in venue.lower():
        arxiv = normalize_arxiv(eprint) or normalize_arxiv(venue)
    arxiv = arxiv or normalize_arxiv(fields.get("url", "")) or normalize_arxiv(venue)
    doi = normalize_doi(fields.get("doi")) or normalize_doi(
        next(iter(re.findall(r"doi\.org/(\S+)", fields.get("url", ""))), None)
    )
    return Reference(
        key=key,
        title=delatex(fields.get("title", "")),
        authors=split_authors(fields.get("author", "")),
        year=_year(fields),
        doi=doi,
        arxiv=arxiv,
        venue=delatex(venue),
        url=fields.get("url", ""),
        raw=raw,
    )


def parse_bibtex(text: str) -> list[Reference]:
    """Every entry of a BibTeX file as a :class:`Reference`."""
    out = []
    for kind, key, fields in parse_entries(text):
        out.append(to_reference(kind, key, fields))
    return out
