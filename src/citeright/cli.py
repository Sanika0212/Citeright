"""Command line: ``citeright check``, ``citeright resolve`` and ``citeright bench``.

citeright check paper.tex --bib refs.bib --html report.html
citeright check draft.md --bib refs.bib --no-support --fail-on retracted,not_found
citeright resolve "Attention is all you need" --author Vaswani --year 2017
citeright bench match
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from citeright import __version__
from citeright.model import Reference, Report, Status, Verdict

_FAIL: dict[str, Callable[[Report], int]] = {
    "retracted": lambda r: r.counts()["retracted"],
    "not_found": lambda r: r.counts()["not_found"] + len(r.unknown_keys),
    "mismatch": lambda r: r.counts()["mismatch"],
    "contradicts": lambda r: r.counts()["claims_contradicts"],
    "unsupported": lambda r: r.counts()["claims_not_found"],
}
_ICON = {
    Status.VERIFIED: "ok ",
    Status.MISMATCH: "!! ",
    Status.NOT_FOUND: "?? ",
    Status.RETRACTED: "XX ",
    Status.UNCHECKED: ".. ",
}


def _print_report(report: Report) -> None:
    for r in report.resolutions:
        print(f"{_ICON[r.status]}{r.reference.key:24s} {r.status.value}")
        for issue in r.issues:
            print(f"     {issue}")
    for key in report.unknown_keys:
        print(f"?? {key:24s} cited but not in the bibliography")
    shown = [c for c in report.checks if c.verdict in (Verdict.CONTRADICTS, Verdict.NOT_FOUND)]
    if shown:
        print()
    for c in shown:
        print(
            f"line {c.citation.line}: [{c.key}] {c.verdict.value} (P(supports) {c.probability:.2f})"
        )
        print(f"     {c.citation.claim}")
    counts = report.counts()
    print()
    print(
        f"{len(report.resolutions)} references: {counts['verified']} verified, "
        f"{counts['mismatch']} mismatched, {counts['not_found']} not found, "
        f"{counts['retracted']} retracted"
        + (f", {counts['unchecked']} unchecked" if counts["unchecked"] else "")
    )
    if report.checks:
        print(
            f"{len(report.checks)} citations checked: {counts['claims_supports']} supported, "
            f"{counts['claims_contradicts']} contradicted, {counts['claims_not_found']} not "
            f"found in the cited work, {counts['claims_no_text']} without text"
        )


def check_command(args: argparse.Namespace) -> int:
    from citeright.check import check
    from citeright.report import write_report
    from citeright.support import SupportChecker

    failing = [f for f in args.fail_on.split(",") if f] if args.fail_on else []
    unknown = [f for f in failing if f not in _FAIL]
    if unknown:
        print(f"unknown --fail-on values: {unknown}; choose from {sorted(_FAIL)}")
        return 2
    draft = Path(args.draft).read_text(encoding="utf-8")
    bib = Path(args.bib).read_text(encoding="utf-8") if args.bib else None
    texts = {}
    if args.texts:
        for path in Path(args.texts).glob("*.txt"):
            texts[path.stem] = path.read_text(encoding="utf-8")
    support: SupportChecker | bool = False if args.no_support else SupportChecker(args.checker)
    report = check(draft, bib, style=args.style, support=support, texts=texts)
    _print_report(report)
    if args.html:
        write_report(report, args.html, title=f"Citation check: {Path(args.draft).name}")
        print(f"report written to {args.html}")
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1))
    return 1 if any(_FAIL[f](report) for f in failing) else 0


def resolve_command(args: argparse.Namespace) -> int:
    from citeright.resolve import Resolver

    ref = Reference(
        key="query",
        title=args.title,
        authors=args.author or [],
        year=args.year,
        doi=args.doi,
        arxiv=args.arxiv,
    )
    r = Resolver().resolve(ref)
    print(json.dumps(r.to_dict(), indent=1, default=str))
    return 0 if r.status is Status.VERIFIED else 1


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["bench"]:
        from citeright.bench import run

        if len(argv) < 2:
            print("usage: citeright bench {match,support} [options]")
            return 2
        return run(argv[1], argv[2:])
    parser = argparse.ArgumentParser(prog="citeright", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"citeright {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="check every reference and citing sentence of a draft")
    c.add_argument("draft", help="LaTeX, Markdown or plain text")
    c.add_argument("--bib", help="BibTeX file or plain-text reference list")
    c.add_argument(
        "--style", default="auto", choices=["auto", "latex", "pandoc", "numeric", "author-year"]
    )
    c.add_argument("--checker", default="nli", help="nli, minicheck, embedding or lexical")
    c.add_argument("--no-support", action="store_true", help="only check the references")
    c.add_argument("--texts", help="folder of <key>.txt full texts to check claims against")
    c.add_argument("--html", help="write a single-file HTML report")
    c.add_argument("--json", help="write the findings as JSON")
    c.add_argument(
        "--fail-on",
        default="",
        help="exit 1 if any of these are found (comma-separated): " + ", ".join(sorted(_FAIL)),
    )
    c.set_defaults(func=check_command)
    r = sub.add_parser("resolve", help="look up one reference")
    r.add_argument("title")
    r.add_argument("--author", action="append")
    r.add_argument("--year", type=int)
    r.add_argument("--doi")
    r.add_argument("--arxiv")
    r.set_defaults(func=resolve_command)
    sub.add_parser("bench", help="run a benchmark suite: match or support")
    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
