"""Single-file HTML report of a citation check.

No server, build step or network: styles, data and a few lines of JavaScript are inline.
Every status carries an icon and a label as well as a colour. The page shows a summary,
a graph of which sentences cite which references, every reference with what was wrong
with it, and every citing sentence with the passage of the cited work that decides it.
It follows the system light/dark preference.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from citeright import __version__
from citeright.model import Report, Resolution, Status, SupportCheck, Verdict

STATUS = {
    Status.VERIFIED: ("✓", "Verified", "good"),
    Status.MISMATCH: ("!", "Mismatch", "warn"),
    Status.NOT_FOUND: ("?", "Not found", "bad"),
    Status.RETRACTED: ("⊘", "Retracted", "bad"),
    Status.UNCHECKED: ("…", "Unchecked", "muted"),
}
VERDICT = {
    Verdict.SUPPORTS: ("✓", "Supports", "good"),
    Verdict.CONTRADICTS: ("✗", "Contradicts", "bad"),
    Verdict.NOT_FOUND: ("?", "Not found in the cited work", "warn"),
    Verdict.NO_TEXT: ("–", "No text to check", "muted"),
}

_CSS = """
:root { color-scheme: light;
  --bg:#f9f9f7; --surface:#fcfcfb; --raised:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --good:#0ca30c; --warn:#c98500; --bad:#d03b3b; --accent:#2a78d6;
  --good-bg:rgba(12,163,12,.10); --warn-bg:rgba(250,178,25,.20); --bad-bg:rgba(208,59,59,.12); --muted-bg:rgba(137,135,129,.14); }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --bg:#0d0d0d; --surface:#1a1a19; --raised:#222220; --ink:#ffffff; --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5;
  --warn:#eda100; --good-bg:rgba(12,163,12,.20); --warn-bg:rgba(250,178,25,.18); --bad-bg:rgba(208,59,59,.26); --muted-bg:rgba(137,135,129,.22); } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19; --raised:#222220; --ink:#ffffff;
  --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; --warn:#eda100; --good-bg:rgba(12,163,12,.20);
  --warn-bg:rgba(250,178,25,.18); --bad-bg:rgba(208,59,59,.26); --muted-bg:rgba(137,135,129,.22); }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main, header { max-width:1100px; margin:0 auto; padding:0 20px; }
header { padding-top:22px; }
h1 { font-size:21px; margin:0 0 4px; } h2 { font-size:16px; margin:26px 0 10px; }
.sub { color:var(--ink2); font-size:13.5px; }
.stats { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 4px; }
.chip { display:inline-flex; gap:6px; align-items:center; border:1px solid var(--line); background:var(--surface);
  border-radius:999px; padding:2px 10px; font-size:13px; white-space:nowrap; }
.badge { display:inline-flex; gap:5px; align-items:center; border-radius:6px; padding:1px 8px; font-size:12.5px; font-weight:600; }
.good { color:var(--good); background:var(--good-bg); } .warn { color:var(--warn); background:var(--warn-bg); }
.bad { color:var(--bad); background:var(--bad-bg); } .muted { color:var(--ink2); background:var(--muted-bg); }
.card { background:var(--raised); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:10px 0; }
.card .top { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.key { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; color:var(--ink2); }
.ref { margin:6px 0 0; } .rec { margin:4px 0 0; color:var(--ink2); font-size:13.5px; }
ul.issues { margin:8px 0 0; padding-left:20px; } ul.issues li { margin:2px 0; }
blockquote { margin:8px 0 0; padding:6px 10px; border-left:3px solid var(--line); color:var(--ink2); font-size:13.5px; }
.claim { font-size:15px; } .line { color:var(--muted); font-size:12.5px; }
.toolbar { display:flex; gap:8px; margin:6px 0 0; }
button { font:inherit; font-size:13px; border:1px solid var(--line); background:var(--surface); color:var(--ink);
  border-radius:999px; padding:3px 12px; cursor:pointer; }
button[aria-pressed="true"] { border-color:var(--accent); color:var(--accent); }
.graph { overflow-x:auto; background:var(--raised); border:1px solid var(--line); border-radius:10px; padding:8px; }
.graph svg text { fill:var(--ink2); font-size:11px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
a { color:var(--accent); } footer { color:var(--muted); font-size:12.5px; padding:30px 20px; text-align:center; }
.hidden { display:none; }
"""

_JS = """
const buttons = document.querySelectorAll('[data-filter]');
buttons.forEach(b => b.addEventListener('click', () => {
  buttons.forEach(o => o.setAttribute('aria-pressed', String(o === b)));
  const problems = b.dataset.filter === 'problems';
  document.querySelectorAll('[data-ok]').forEach(el =>
    el.classList.toggle('hidden', problems && el.dataset.ok === 'true'));
}));
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _badge(icon: str, label: str, tone: str) -> str:
    return f'<span class="badge {tone}"><span aria-hidden="true">{icon}</span>{_e(label)}</span>'


def _doi_link(doi: str | None) -> str:
    return f'<a href="https://doi.org/{_e(doi)}">{_e(doi)}</a>' if doi else ""


def _resolution(r: Resolution) -> str:
    ref, rec = r.reference, r.record
    icon, label, tone = STATUS[r.status]
    written = ", ".join(
        x
        for x in (
            "; ".join(ref.authors[:4]) + (" et al." if len(ref.authors) > 4 else ""),
            f"<i>{_e(ref.title)}</i>" if ref.title else _e(ref.raw[:200]),
            _e(ref.venue),
            _e(ref.year or ""),
            _doi_link(ref.doi),
        )
        if x
    )
    found = ""
    if rec is not None and r.status is not Status.NOT_FOUND:
        names = ", ".join(rec.authors[:4]) + (" et al." if len(rec.authors) > 4 else "")
        found = (
            f'<p class="rec">Found ({_e(rec.source)}, match {r.probability:.0%}): '
            f"{_e(names)}, <i>{_e(rec.title)}</i>, {_e(rec.venue)} {_e(rec.year or '')} "
            f"{_doi_link(rec.doi)}</p>"
        )
    elif rec is not None:
        found = (
            f'<p class="rec">Closest record (match {r.probability:.0%}): <i>{_e(rec.title)}</i> '
            f"{_e(rec.year or '')}</p>"
        )
    issues = "".join(f"<li>{_e(i)}</li>" for i in r.issues)
    ok = r.status is Status.VERIFIED and not r.issues
    return (
        f'<div class="card" data-ok="{str(ok).lower()}" id="ref-{_e(ref.key)}"><div class="top">'
        f'{_badge(icon, label, tone)}<span class="key">{_e(ref.key)}</span></div>'
        f'<p class="ref">{written}</p>{found}'
        + (f'<ul class="issues">{issues}</ul>' if issues else "")
        + "</div>"
    )


def _claims(checks: list[SupportCheck]) -> str:
    groups: dict[int, list[SupportCheck]] = {}
    for c in checks:
        groups.setdefault(id(c.citation), []).append(c)
    out = []
    for group in groups.values():
        citation = group[0].citation
        rows = []
        for c in group:
            icon, label, tone = VERDICT[c.verdict]
            detail = ""
            if c.verdict is not Verdict.NO_TEXT:
                detail = (
                    f' <span class="sub">P(supports) {c.probability:.0%}, from the '
                    f"{_e(c.text_source)}</span>"
                    + (f"<blockquote>{_e(c.evidence)}</blockquote>" if c.evidence else "")
                )
            rows.append(
                f'<div style="margin-top:8px">{_badge(icon, label, tone)} '
                f'<a class="key" href="#ref-{_e(c.key)}">{_e(c.key)}</a>{detail}</div>'
            )
        ok = all(c.verdict in (Verdict.SUPPORTS, Verdict.NO_TEXT) for c in group)
        out.append(
            f'<div class="card" data-ok="{str(ok).lower()}"><div class="claim">{_e(citation.claim)}</div>'
            f'<div class="line">line {citation.line}</div>{"".join(rows)}</div>'
        )
    return "".join(out)


def _graph(report: Report) -> str:
    """Sentences on the left, references on the right; edges coloured by verdict."""
    refs = [r.reference.key for r in report.resolutions]
    sentences: list[int] = []
    for c in report.checks:
        if id(c.citation) not in sentences:
            sentences.append(id(c.citation))
    if not refs or not sentences:
        return ""
    gap = 26
    height = max(len(refs), len(sentences)) * gap + 30
    width = 640
    ry = {k: 20 + i * height / max(len(refs), 1) for i, k in enumerate(refs)}
    sy = {s: 20 + i * height / max(len(sentences), 1) for i, s in enumerate(sentences)}
    tone = {
        "good": "var(--good)",
        "warn": "var(--warn)",
        "bad": "var(--bad)",
        "muted": "var(--muted)",
    }
    parts = []
    for c in report.checks:
        if c.key not in ry:
            continue
        color = tone[VERDICT[c.verdict][2]]
        dash = ' stroke-dasharray="4 3"' if c.verdict is Verdict.NO_TEXT else ""
        parts.append(
            f'<path d="M 70 {sy[id(c.citation)]} C 300 {sy[id(c.citation)]}, 340 {ry[c.key]}, '
            f'{width - 150} {ry[c.key]}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>'
        )
    for i, s in enumerate(sentences):
        parts.append(f'<circle cx="62" cy="{sy[s]}" r="6" fill="var(--accent)"/>')
        parts.append(f'<text x="8" y="{sy[s] + 4}">S{i + 1}</text>')
    for r in report.resolutions:
        color = tone[STATUS[r.status][2]]
        y = ry[r.reference.key]
        parts.append(
            f'<rect x="{width - 150}" y="{y - 7}" width="14" height="14" rx="3" fill="{color}"/>'
        )
        parts.append(f'<text x="{width - 130}" y="{y + 4}">{_e(r.reference.key[:18])}</text>')
    return (
        f'<div class="graph"><svg role="img" aria-label="Which sentences cite which references" '
        f'width="{width}" height="{height + 10}" viewBox="0 0 {width} {height + 10}">{"".join(parts)}</svg></div>'
    )


def render(report: Report, *, title: str = "Citation check") -> str:
    counts = report.counts()
    chips = []
    for status, (icon, label, tone) in STATUS.items():
        if counts.get(status.value):
            chips.append(
                f'<span class="chip">{_badge(icon, label, tone)} {counts[status.value]}</span>'
            )
    for verdict, (icon, label, tone) in VERDICT.items():
        n = counts.get(f"claims_{verdict.value}", 0)
        if n:
            chips.append(f'<span class="chip">{_badge(icon, label, tone)} {n} claims</span>')
    if report.unknown_keys:
        chips.append(
            f'<span class="chip">{_badge("?", "Cited, not in bibliography", "bad")} {len(report.unknown_keys)}</span>'
        )
    problems = report.problems()
    summary = (
        f"{len(report.resolutions)} references, {len({id(c.citation) for c in report.checks})} citing "
        f"sentences checked; {problems} problem{'s' if problems != 1 else ''} found."
    )
    unknown = (
        "<h2>Cited but missing from the bibliography</h2><p>"
        + ", ".join(f'<span class="key">{_e(k)}</span>' for k in report.unknown_keys)
        + "</p>"
        if report.unknown_keys
        else ""
    )
    uncited = (
        "<p class='sub'>In the bibliography but never cited: "
        + ", ".join(f'<span class="key">{_e(k)}</span>' for k in report.uncited)
        + "</p>"
        if report.uncited
        else ""
    )
    graph = _graph(report)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head><body>
<header><h1>{_e(title)}</h1><div class="sub">{_e(summary)}</div>
<div class="stats">{"".join(chips)}</div>
<div class="toolbar"><button data-filter="all" aria-pressed="true">Everything</button>
<button data-filter="problems" aria-pressed="false">Problems only</button></div></header>
<main>{unknown}
{"<h2>Who cites what</h2>" + graph if graph else ""}
<h2>Citing sentences</h2>{_claims(report.checks) or "<p class='sub'>No claims were checked.</p>"}
<h2>References</h2>{"".join(_resolution(r) for r in report.resolutions)}{uncited}</main>
<footer>Citeright {_e(__version__)}. Metadata from doi.org, Crossref, OpenAlex and arXiv.</footer>
<script>{_JS}</script></body></html>
"""


def write_report(report: Report, path: str | Path, *, title: str = "Citation check") -> Path:
    out = Path(path)
    out.write_text(render(report, title=title), encoding="utf-8")
    return out
