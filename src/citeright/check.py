"""Check a whole draft: parse, resolve every reference, check every citing sentence.

.. code-block:: python

    from citeright.check import check

    report = check(open("paper.tex").read(), open("refs.bib").read())
    for r in report.resolutions:
        print(r.reference.key, r.status.value, r.issues)
    for c in report.checks:
        print(c.verdict.value, c.citation.claim, "<-", c.key, c.evidence)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from citeright.bibtex import parse_bibtex
from citeright.citations import (
    body_and_references,
    extract_citations,
    match_author_year,
    parse_reference_list,
    parse_reference_string,
)
from citeright.model import Reference, Report, SupportCheck, Verdict
from citeright.resolve import Resolver
from citeright.support import SupportChecker

_BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}")


def read_bibliography(
    draft: str, bibliography: str | Sequence[Reference] | None
) -> list[Reference]:
    """References from a BibTeX string, a list, a ``thebibliography`` block or a reference list."""
    if bibliography is not None and not isinstance(bibliography, str):
        return list(bibliography)
    if bibliography and "@" in bibliography and re.search(r"@\w+\s*[{(]", bibliography):
        return parse_bibtex(bibliography)
    if bibliography:
        return parse_reference_list(bibliography)
    items = list(_BIBITEM.finditer(draft))
    if items:
        out = []
        for i, m in enumerate(items):
            end = items[i + 1].start() if i + 1 < len(items) else len(draft)
            chunk = re.split(r"\\end\{thebibliography\}", draft[m.end() : end])[0]
            out.append(parse_reference_string(m.group(1), " ".join(chunk.split())))
        return out
    _, refs = body_and_references(draft)
    return parse_reference_list(refs) if refs.strip() else []


def check(
    draft: str,
    bibliography: str | Sequence[Reference] | None = None,
    *,
    style: str = "auto",
    resolver: Resolver | None = None,
    support: SupportChecker | bool | None = True,
    texts: Mapping[str, str] | None = None,
    workers: int = 4,
) -> Report:
    """Resolve every reference and check every citing sentence of ``draft``.

    ``texts`` maps reference keys to full text you have (the abstract from the
    metadata sources is used otherwise). ``support=False`` skips claim checking.
    """
    references = read_bibliography(draft, bibliography)
    citations = extract_citations(draft, style=style)
    by_key = {r.key: r for r in references}
    for citation in citations:  # author-year keys are resolved against the list
        citation.keys = [
            k if k in by_key else (getattr(match_author_year(k, references), "key", None) or k)
            for k in citation.keys
        ]
    cited = {k for c in citations for k in c.keys}
    resolver = resolver or Resolver()
    resolutions = resolver.resolve_all(references, workers=workers)
    checks: list[SupportCheck] = []
    if support is not False and citations:
        checker = support if isinstance(support, SupportChecker) else SupportChecker()
        records = {r.reference.key: r.record for r in resolutions}
        for citation in citations:
            for key in citation.keys:
                if key not in by_key:
                    continue
                full = (texts or {}).get(key, "")
                record = records.get(key)
                text = full or (record.abstract if record else "")
                source = "full text" if full else "abstract" if text else ""
                if not text:
                    checks.append(SupportCheck(citation, key, Verdict.NO_TEXT))
                    continue
                checks.append(checker.check(citation, key, text, source))
    return Report(
        resolutions=resolutions,
        checks=checks,
        unknown_keys=sorted(cited - set(by_key)),
        uncited=[r.key for r in references if r.key not in cited] if citations else [],
    )
