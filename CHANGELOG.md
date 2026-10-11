# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-12

### Added

1. Parsing of LaTeX, Pandoc Markdown, numbered and author-year citations, with BibTeX,
   `thebibliography` blocks and plain-text reference lists (IEEE, APA, ACL, Vancouver).
2. Resolution of every reference against the DOI handle system, Crossref, OpenAlex and
   arXiv, with a fitted match model and statuses verified, mismatch, not found,
   retracted and unchecked; every mismatch says which field is wrong and what the
   record says.
3. Retractions, corrections and expressions of concern from Crossref (Retraction Watch
   data) and OpenAlex.
4. Support checking of each citing sentence against the cited abstract or full text,
   with NLI, MiniCheck, embedding and word-overlap checkers and calibrated three-way
   verdicts.
5. A single-file HTML report, a JSON export, and `citeright check --fail-on` for CI.
6. Benchmarks: `citeright bench match` (real references with planted faults) and
   `citeright bench support` (SciFact and Citation-Integrity).
7. An on-disk HTTP cache, request spacing, retries, and graceful handling of the
   OpenAlex daily budget.
